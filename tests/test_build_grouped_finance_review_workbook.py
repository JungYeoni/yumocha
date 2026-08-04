from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule

from scripts.build_grouped_finance_review_workbook import _copy_conditional_formatting


def test_copy_conditional_formatting_preserves_each_original_scope():
    workbook = Workbook()
    source = workbook.active
    target = workbook.create_sheet("target")
    source.conditional_formatting.add(
        "A2:C10",
        FormulaRule(formula=["$A2=1"]),
    )
    source.conditional_formatting.add(
        "D2:D5",
        FormulaRule(formula=['$D2="확정"']),
    )

    _copy_conditional_formatting(source, target)

    copied = list(target.conditional_formatting)
    assert [str(item.sqref) for item in copied] == ["A2:C10", "D2:D5"]
    assert [item.rules[0].formula for item in copied] == [["$A2=1"], ['$D2="확정"']]
