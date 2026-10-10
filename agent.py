
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../SCSE_26_GEX3/models/qwen3-0.6b"

print("Loading Tokenizer and Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype="auto"
)

model.eval()


def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return response.strip()


def clean_answer(text):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"^(Assistant|Answer|Draft)\s*:\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    text = re.split(
        r"(?m)^\s*(?:---+|Answer:|Assistant:|Review:|User:|Question:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    return " ".join(text.split()).strip()


def asks_for_phone_number(question):
    question = question.lower()

    return (
        "phone number" in question
        or "telephone number" in question
        or "contact number" in question
        or ("call" in question and "number" in question)
    )


def evidence_has_phone_number(context):
    phone_pattern = re.compile(
        r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)"
    )

    for line in context.splitlines():
        if not re.search(
            r"\b(phone|telephone|tel|hotline|call)\b",
            line,
            flags=re.IGNORECASE
        ):
            continue

        for match in phone_pattern.findall(line):
            digits = re.sub(r"\D", "", match)

            if len(digits) >= 7:
                return True

    return False


def asks_for_location(question):
    question = question.lower()

    return bool(
        re.search(
            r"\b(where|location|located|address|which building|which room)\b",
            question
        )
    )


def evidence_has_location(context):
    patterns = [
        r"\b(?:located|situated|address)\b",
        r"\b(?:building|room|floor|hall|library|centre|center|block|wing)\b",
        r"\b(?:north|south|east|west)\s+campus\b",
        r"\b\d+\s+[\w\s]+\s+(?:street|road|avenue|lane)\b"
    ]

    return any(
        re.search(pattern, context, flags=re.IGNORECASE)
        for pattern in patterns
    )


def validate_evidence(question, context):
    if asks_for_phone_number(question):
        if not evidence_has_phone_number(context):
            return (
                "I cannot provide the exact phone number "
                "because it is not specified in the "
                "retrieved university IT information."
            )

    if asks_for_location(question):
        if not evidence_has_location(context):
            return (
                "The retrieved university IT information "
                "does not specify the requested location."
            )

    return None


def run_agent(question, context):
    print("\n[Agent] Phase 1: Drafting")

    draft_prompt = (
        "You are a university IT support assistant.\n"
        "Answer the user's question using ONLY the evidence below.\n"
        "Do not invent facts, locations, phone numbers, or procedures.\n"
        "If the evidence does not contain the requested information, "
        "clearly say that it is not specified.\n"
        "Give one short, direct answer.\n"
        "Do not add unrelated advice or repeated sections.\n\n"
        f"Evidence:\n{context}\n\n"
        f"User: {question}\n"
        "Assistant:"
    )

    draft_raw = generate_response(
        draft_prompt,
        max_new_tokens=80
    )

    draft = clean_answer(draft_raw)
    print(f"Draft: {draft}")

    print("\n[Agent] Phase 2: Reflecting")

    review_prompt = (
        "You are a strict evidence reviewer.\n"
        "Check if the answer is directly supported by the evidence.\n"
        "Reject fabricated locations, phone numbers, unsupported facts, "
        "and irrelevant instructions.\n"
        "An answer that admits the evidence lacks the requested "
        "information is acceptable.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{context}\n"
        f"Draft: {draft}\n\n"
        "Reply with exactly PASS or REVISE.\n"
        "Review:"
    )

    review_raw = generate_response(
        review_prompt,
        max_new_tokens=12
    )

    match = re.match(
        r"^\s*(PASS|REVISE)\b",
        review_raw,
        flags=re.IGNORECASE
    )

    review = match.group(1).upper() if match else "REVISE"
    print(f"Review: {review_raw}")

    if review == "PASS":
        final_answer = draft

    else:
        print("\n[Agent] Phase 3: Revising")

        revise_prompt = (
            "You are a careful university IT support assistant.\n"
            "The previous answer was rejected.\n"
            "Use only facts explicitly supported by the evidence.\n"
            "If the requested fact is missing, acknowledge that "
            "the provided information does not specify it.\n"
            "Never invent locations, numbers, or extra instructions.\n"
            "Write only the corrected answer.\n\n"
            f"Evidence:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Previous answer: {draft}\n\n"
            "Corrected answer:"
        )

        revised_raw = generate_response(
            revise_prompt,
            max_new_tokens=80
        )

        final_answer = clean_answer(revised_raw)

    print("\n[Agent] Phase 4: Evidence Validation")

    validated_answer = validate_evidence(
        question,
        context
    )

    if validated_answer is not None:
        print(
            "[Agent] The requested fact is not "
            "explicitly supported by the retrieved evidence."
        )

        final_answer = validated_answer

    if not final_answer:
        final_answer = (
            "The retrieved university IT information "
            "does not specify the requested information."
        )

    return final_answer.strip()



