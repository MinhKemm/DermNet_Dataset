"""Compose the 5 x 4 Vietnamese QA standard offline; never calls a model.

Usage: python -m Phase_2.qa_prompts catalog --profile all
An image path in a rendered prompt is not an attached image. The consuming
multimodal runner must attach it separately and own scheduling/checkpoints.
"""

import argparse
import csv
import io
import json
import sys
import unicodedata
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config" / "qa_standard_vi.json"
ATTRIBUTE_FIELDS = ("Size", "Color", "Boundary", "Shape", "Quantity", "Distribution")
LABEL_FIELDS = ("Category", "Location") + ATTRIBUTE_FIELDS


def load_standard():
    """Read the versioned, UTF-8 prompt configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _selection(standard, category, question_type, profile):
    for value, section in ((category, "tasks"), (question_type, "formats"),
                           (profile, "profiles")):
        if not isinstance(value, str) or value not in standard[section]:
            raise ValueError(f"Unknown {section}: {value!r}")


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def validate_context(context, category, attribute_field=None):
    """Validate generation inputs. Returns the requested canonical field.

    The caller supplies a reviewed, versioned label dictionary. This function
    checks its shape, not its clinical correctness or the image attachment.
    """
    standard = load_standard()
    if not isinstance(category, str) or category not in standard["tasks"]:
        raise ValueError(f"Unknown task: {category!r}")
    if not isinstance(context, dict) or not _nonempty_string(context.get("image_id")):
        raise ValueError("context.image_id must be a nonempty string")
    taxonomy = context.get("taxonomy")
    if not isinstance(taxonomy, dict) or not _nonempty_string(taxonomy.get("version")):
        raise ValueError("context.taxonomy.version must be a nonempty string")
    limit = context.get("max_qa", 1)
    if type(limit) is not int or limit < 1:
        raise ValueError("max_qa must be a positive integer")
    field = standard["tasks"][category]["field"]
    if category == "Attribute_Recognition":
        if attribute_field not in ATTRIBUTE_FIELDS:
            raise ValueError("attribute_field must be one of the six attributes")
        field = attribute_field
    elif attribute_field is not None:
        raise ValueError("attribute_field is only valid for Attribute_Recognition")
    fields = taxonomy.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("taxonomy.fields must be an object")
    annotations = context.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("annotations must contain normalized field arrays")
    for key in LABEL_FIELDS + ("visual_evidence",):
        if key in annotations and (not isinstance(annotations[key], list)
                                  or not all(_nonempty_string(item) for item in annotations[key])):
            raise ValueError(f"annotations.{key} must be a list of strings")
    if category == "Diagnosis":
        if not _nonempty_string(context.get("diagnosis_label")):
            raise ValueError("missing_diagnosis_label")
    else:
        labels = fields.get(field)
        if (not isinstance(labels, list) or not labels
                or not all(_nonempty_string(label) for label in labels)
                or len(labels) != len(set(labels))):
            raise ValueError(f"taxonomy.fields.{field} must contain unique canonical labels")
    candidates = context.get("diagnosis_candidates", [])
    if (not isinstance(candidates, list)
            or not all(_nonempty_string(label) for label in candidates)
            or len(candidates) != len(set(candidates))):
        raise ValueError("diagnosis_candidates must be a list of unique disease labels")
    if context.get("correct_option_position") not in (None, "A", "B", "C", "D"):
        raise ValueError("correct_option_position must be A-D when supplied")
    if type(context.get("has_marked_region", False)) is not bool:
        raise ValueError("has_marked_region must be boolean")
    used = context.get("used_questions", [])
    if not isinstance(used, list) or not all(_nonempty_string(q) for q in used):
        raise ValueError("used_questions must be a list of nonempty strings")
    return field


def build_prompt(category, question_type, *, profile="runtime", context=None,
                 attribute_field=None):
    """Render one combination. Without context, render a catalog template.

    Bound inputs are JSON data, not interpolated instructions. Disease gold is
    intentionally omitted from tasks other than Diagnosis. Clinical dictionaries
    and observations must still be reviewed by the caller.
    """
    standard = load_standard()
    _selection(standard, category, question_type, profile)
    task = standard["tasks"][category]
    rules = [f"DermNet QA {standard['version']} | {task['display']} | {question_type}"]
    rules.extend(standard["common_rules"])
    rules.extend(task["rules"])
    rules.extend(standard["formats"][question_type]["rules"])
    rules.append(standard["combinations"][category][question_type])
    rules.extend(standard["profiles"][profile])
    rules.append(f"category={category}; type={question_type}; không tự đổi hai trường này.")
    if context is None:
        rules.append("INPUT_JSON và ảnh thật do runner cấp riêng; đây chỉ là mẫu prompt.")
        return "\n\n".join(rules)
    field = validate_context(context, category, attribute_field)
    # Use a bounded allowlist rather than forwarding arbitrary metadata to a model.
    allowed = ("image_id", "taxonomy", "annotations", "max_qa", "used_questions",
               "has_marked_region", "subtype", "target_scope", "correct_option_position",
               "measurement", "quantity_policy", "location_paths", "run_id", "task_id")
    bound = {key: context[key] for key in allowed if key in context}
    bound["taxonomy"] = {
        "version": context["taxonomy"]["version"],
        "fields": {key: value for key, value in context["taxonomy"]["fields"].items()
                   if key in LABEL_FIELDS},
    }
    bound["annotations"] = {key: value for key, value in context.get("annotations", {}).items()
                            if key in LABEL_FIELDS + ("visual_evidence",)}
    bound["max_qa"] = context.get("max_qa", 1)
    bound["requested_category"] = category
    bound["requested_type"] = question_type
    bound["sub_category"] = field
    if category == "Attribute_Recognition":
        bound["attribute_field"] = field
    if category == "Diagnosis":
        bound["diagnosis_label"] = context["diagnosis_label"]
        bound["diagnosis_candidates"] = context.get("diagnosis_candidates", [])
    return "\n\n".join(rules) + "\nINPUT_JSON\n" + json.dumps(bound, ensure_ascii=False)


def catalog(profile="runtime"):
    """Return 20 ready-to-inspect templates for one profile, or 40 for all."""
    standard = load_standard()
    profiles = tuple(standard["profiles"]) if profile == "all" else (profile,)
    if any(name not in standard["profiles"] for name in profiles):
        raise ValueError(f"Unknown profile: {profile!r}")
    return [
        {
            "prompt_id": f"DNQA-{standard['version']}-{task['id']}-{kind['id']}-{name}",
            "category": category, "type": question_type, "profile": name,
            "prompt": build_prompt(category, question_type, profile=name),
        }
        for name in profiles
        for category, task in standard["tasks"].items()
        for question_type, kind in standard["formats"].items()
    ]


def _text_key(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _qa_errors(qa, context, category, question_type, field):
    """Check one candidate structurally; no inference about image truth."""
    if not isinstance(qa, dict):
        return ["must be an object"]
    errors = []
    expected = {"category": category, "type": question_type,
                "sub_category": field, "status": "candidate"}
    for key, value in expected.items():
        if qa.get(key) != value:
            errors.append(f"{key} must be {value!r}")
    for key in ("question", "answer", "answer_label", "target", "scope"):
        if not _nonempty_string(qa.get(key)):
            errors.append(f"{key} must be a nonempty string")
    labels = qa.get("source_labels")
    if (not isinstance(labels, list) or len(labels) != 1
            or not all(_nonempty_string(label) for label in labels)):
        errors.append("source_labels must contain exactly one canonical label")
    evidence = qa.get("evidence")
    if not isinstance(evidence, list) or not all(_nonempty_string(e) for e in evidence):
        errors.append("evidence must be a list of observations")
    elif category != "Diagnosis" and not evidence:
        errors.append("image-based QA needs evidence")
    if not isinstance(qa.get("rationale"), str):
        errors.append("rationale must be a string")
    if errors:
        return errors

    gold = context.get("diagnosis_label") if category == "Diagnosis" else None
    allowed_labels = ([gold] if gold else context["taxonomy"]["fields"][field])
    if labels[0] not in allowed_labels:
        errors.append("source_labels is outside the requested dictionary/source gold")
    if qa["scope"] not in ("whole_image", "described_region", "marked_region"):
        errors.append("scope is invalid")
    if not context.get("has_marked_region", False):
        if qa["scope"] == "marked_region" or any(
                text in qa["question"].casefold() for text in ("vùng đánh dấu", "vùng được đánh dấu")):
            errors.append("marked region was not supplied")

    answer, content = qa["answer"], qa["answer_label"]
    options = qa.get("options")
    if question_type == "Multi_choice":
        if (not isinstance(options, dict) or set(options) != set("ABCD")
                or not all(_nonempty_string(value) for value in options.values())):
            errors.append("options must contain exactly four nonempty A-D choices")
        else:
            if len({_text_key(value) for value in options.values()}) != 4:
                errors.append("options are duplicated")
            if answer not in options or options.get(answer) != content:
                errors.append("answer key does not match answer_label")
            assigned = context.get("correct_option_position")
            if assigned is not None and answer != assigned:
                errors.append("answer key differs from assigned position")
            if category != "Lesion_Reasoning":
                choice_labels = ([gold] + context.get("diagnosis_candidates", [])
                                 if gold else allowed_labels)
                if any(value not in choice_labels for value in options.values()):
                    errors.append("options contain labels outside the requested field/disease candidates")
            elif any(value in allowed_labels for value in options.values()):
                errors.append("Reasoning choices must be explanations, not lesion names")
    else:
        if options != {}:
            errors.append("non-MCQ options must be an empty object")
        if answer != content:
            errors.append("answer must equal answer_label for non-MCQ")
    if question_type == "Judgement":
        if answer not in ("Có", "Không"):
            errors.append("Judgement answer must be Có or Không")
        if gold:
            claim = qa.get("claim_label")
            if not _nonempty_string(claim) or claim not in [gold] + context.get("diagnosis_candidates", []):
                errors.append("claim_label must be a supplied disease label")
            elif answer != ("Có" if claim == gold else "Không"):
                errors.append("diagnosis judgement contradicts given gold")
    elif category != "Lesion_Reasoning" and content != labels[0]:
        errors.append("answer content differs from source label")
    if question_type == "Fill_in_blank" and qa["question"].count("____") != 1:
        errors.append("Fill_in_blank requires exactly one ____")
    if category == "Lesion_Reasoning":
        if not qa["rationale"].strip():
            errors.append("Reasoning needs a short rationale")
        if question_type != "Judgement" and content in allowed_labels:
            errors.append("Reasoning answer must be evidence/explanation, not only a lesion name")
    return errors


def validate_response(response, context, category, question_type, *, attribute_field=None):
    """Return structural errors for untrusted model JSON, not clinical approval.

    Invalid caller context raises ValueError. Invalid model output is reported as
    a list of errors, without coercing labels or silently repairing disease gold.
    """
    standard = load_standard()
    _selection(standard, category, question_type, "runtime")
    field = validate_context(context, category, attribute_field)
    if not isinstance(response, dict):
        return ["response must be an object"]
    errors = []
    metadata = {"schema_version": standard["version"], "image_id": context["image_id"],
                "taxonomy_version": context["taxonomy"]["version"]}
    for key, value in metadata.items():
        if response.get(key) != value:
            errors.append(f"{key} does not match the request")
    for key in ("qas", "skipped", "needs_review"):
        if not isinstance(response.get(key), list):
            errors.append(f"{key} must be a list")
    if any(not isinstance(response.get(key), list) for key in ("qas", "skipped", "needs_review")):
        return errors
    for key in ("skipped", "needs_review"):
        for index, item in enumerate(response[key]):
            if not isinstance(item, dict) or not _nonempty_string(item.get("reason")):
                errors.append(f"{key}[{index}] needs a nonempty reason")
            elif item["reason"] in ("image_unavailable", "image_label_conflict", "missing_diagnosis_label"):
                if response["qas"]:
                    errors.append(f"{key}[{index}]: blocking reason cannot coexist with qas")
    if response["qas"] and category == "Attribute_Recognition" and field in ("Size", "Quantity"):
        support_key = "measurement" if field == "Size" else "quantity_policy"
        support = context.get(support_key)
        if (not isinstance(support, dict) or support.get("reviewed") is not True
                or support.get("visible_to_answerer") is not True
                or not _nonempty_string(support.get("description"))):
            errors.append(f"{field} needs reviewed, answerer-visible {support_key}")
    if len(response["qas"]) > context.get("max_qa", 1):
        errors.append("qas exceeds max_qa")
    if not any(response[key] for key in ("qas", "skipped", "needs_review")):
        errors.append("empty qas needs an explained skip or review")
    seen = {_text_key(question) for question in context.get("used_questions", [])}
    for index, qa in enumerate(response["qas"]):
        errors.extend(f"qas[{index}]: {error}" for error in
                      _qa_errors(qa, context, category, question_type, field))
        if isinstance(qa, dict) and _nonempty_string(qa.get("question")):
            key = _text_key(qa["question"])
            if key in seen:
                errors.append(f"qas[{index}]: duplicate/previously used question")
            seen.add(key)
    return errors


def to_tsv(response, context, category, question_type, *, attribute_field=None, start_index=0):
    """Export structurally valid candidates to a NEW staging TSV string.

    This keeps the old six columns/types, not the old category grouping. Calling
    it is not clinical approval or automatic publication into a benchmark.
    """
    errors = validate_response(response, context, category, question_type,
                               attribute_field=attribute_field)
    if errors:
        raise ValueError("; ".join(errors))
    if not _nonempty_string(context.get("image_path")):
        raise ValueError("image_path is required for TSV export")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("index", "image_path", "question", "answer", "category", "type"))
    for index, qa in enumerate(response["qas"], start=start_index):
        question = qa["question"]
        if qa["type"] == "Multi_choice":
            question += "\n" + "\n".join(f"{key}. {qa['options'][key]}" for key in "ABCD")
        writer.writerow((index, context["image_path"], question, qa["answer"], qa["category"], qa["type"]))
    return stream.getvalue()


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _write_output(content, output):
    if output:
        # Exclusive creation protects existing artifacts/datasets by default.
        with Path(output).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content + "\n")
    else:
        sys.stdout.write(content + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("catalog", help="Export 20/40 templates without calling a model")
    export.add_argument("--profile", choices=("paper", "runtime", "all"), default="all")
    export.add_argument("--output")
    render = commands.add_parser("render", help="Bind inputs; attach the image in your runner separately")
    render.add_argument("--task", required=True)
    render.add_argument("--type", required=True)
    render.add_argument("--profile", choices=("paper", "runtime"), default="runtime")
    render.add_argument("--context", required=True)
    render.add_argument("--attribute")
    render.add_argument("--output")
    for name in ("validate", "to-tsv"):
        check = commands.add_parser(name, help="Check structure; never verifies clinical image truth")
        check.add_argument("--task", required=True)
        check.add_argument("--type", required=True)
        check.add_argument("--context", required=True)
        check.add_argument("--response", required=True)
        check.add_argument("--attribute")
        check.add_argument("--output")
        if name == "to-tsv":
            check.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            content = json.dumps(catalog(args.profile), ensure_ascii=False, indent=2)
        elif args.command == "render":
            content = build_prompt(args.task, args.type, profile=args.profile,
                                   context=_read_json(args.context), attribute_field=args.attribute)
        else:
            context = _read_json(args.context)
            response = _read_json(args.response)
            if args.command == "to-tsv":
                content = to_tsv(response, context, args.task, args.type,
                                 attribute_field=args.attribute, start_index=args.start_index)
            else:
                errors = validate_response(response, context, args.task, args.type,
                                           attribute_field=args.attribute)
                content = json.dumps({"structurally_valid": not errors,
                                      "clinical_validation": "not_performed", "errors": errors},
                                     ensure_ascii=False, indent=2)
                _write_output(content, args.output)
                return 1 if errors else 0
        _write_output(content, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
