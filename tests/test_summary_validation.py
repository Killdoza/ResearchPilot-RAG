from researchpilot.summary_validation import validate_and_repair_summary


def test_summary_validator_adds_missing_sections_without_second_generation():
    result = validate_and_repair_summary("## 核心方法\n\n使用图索引。[证据 1]", intent="overview")
    assert result.repaired
    assert "## 核心方法" in result.text
    assert "## 主要结果" in result.text
    assert "当前检索证据不足" in result.text
    assert "missing_section:主要结果" in result.issues


def test_summary_validator_removes_empty_table_and_uncited_facts():
    result = validate_and_repair_summary(
        "## 实验结果\n\n| 指标 | 数值 |\n|---|---|\n\n## 结论与局限\n\n性能最好。",
        intent="overview",
    )
    assert "|---|---|" not in result.text
    assert "性能最好" not in result.text
    assert "empty_table:实验结果" in result.issues
    assert any(issue.startswith("uncited_line:结论与局限") for issue in result.issues)


def test_summary_validator_removes_later_empty_table_but_preserves_first_table():
    text = """## 实验结果

| 数据集 | 胜率 |
|---|---|
| Legal | 85.6% | [证据 1]

第二组结果。[证据 2]

| 数据集 | 胜率 |
|---|---|

## 结论

方法有效。[证据 2]
"""
    result = validate_and_repair_summary(text, intent="motivation_to_results")

    assert "| Legal | 85.6% | [证据 1]" in result.text
    assert result.text.count("|---|---|") == 1
    assert "empty_table:实验结果" in result.issues


def test_summary_validator_preserves_cited_section():
    result = validate_and_repair_summary(
        "## 核心贡献\n\n- 提出轻量检索方法。[证据 2]", intent="contributions_and_limitations"
    )
    assert "提出轻量检索方法" in result.text


def test_summary_validator_accepts_equivalent_headings():
    text = """## 研究问题

问题。[证据 1]

## 核心方法

方法。[证据 1]

## 实验与结果

设置与结果。[证据 2]

## 结论

结论。[证据 2]
"""
    result = validate_and_repair_summary(text, intent="motivation_to_results")

    assert "missing_section:研究动机" not in result.issues
    assert "missing_section:实验设置" not in result.issues
    assert "missing_section:实验结果" not in result.issues


def test_summary_validator_propagates_table_citation_to_each_data_row():
    result = validate_and_repair_summary(
        "## 实验结果\n\n表格如下。[证据 2]\n\n| 数据集 | 胜率 |\n|---|---|\n| Legal | 85.6% |",
        intent="overview",
    )

    assert "| Legal | 85.6% | [证据 2]" in result.text


def test_summary_validator_removes_uncited_bullet_but_keeps_cited_fact():
    result = validate_and_repair_summary(
        "## 核心方法\n\n- 无法追溯的方法断言。\n- 图索引。[证据 1]",
        intent="overview",
    )

    assert "无法追溯的方法断言" not in result.text
    assert "图索引。[证据 1]" in result.text
    assert any(issue.startswith("uncited_line:核心方法") for issue in result.issues)
