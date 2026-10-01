"""Compose the 5 x 4 Vietnamese QA standard offline; never calls a model.

Usage: python -m Phase_2.qa_prompts catalog --profile all
An image path in a rendered prompt is not an attached image. The consuming
multimodal runner must attach it separately and own scheduling/checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config" / "qa_standard_vi.json"
ATTRIBUTE_FIELDS = ("Size", "Color", "Boundary", "Shape", "Quantity", "Distribution")


def load_standard():
    """Read the versioned, UTF-8 prompt configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _selection(standard, category, question_type, profile):
    for value, section in ((category, "tasks"), (question_type, "formats"),
                           (profile, "profiles")):
        if value not in standard[section]:
            raise ValueError(f"Unknown {section}: {value!r}")


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def validate_context(context, category, attribute_field=None):
    """Validate generation inputs. Returns the requested canonical field.

    The caller supplies a reviewed, versioned label dictionary. This function
    checks its shape, not its clinical correctness or the image attachment.
    """
    standard = load_standard()
    if category not in standard["tasks"]:
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
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            content = json.dumps(catalog(args.profile), ensure_ascii=False, indent=2)
        else:
            content = build_prompt(args.task, args.type, profile=args.profile,
                                   context=_read_json(args.context), attribute_field=args.attribute)
        _write_output(content, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
