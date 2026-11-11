
# 环境自检工具

本项目提供一个轻量级环境自检脚本，帮助用户在安装完整依赖之前检查基础运行条件。

## 1. 功能

目前支持以下检查：

- 检测当前 Python 主版本和次版本是否为 3.13。
- 检查项目根目录是否存在 requirements.txt。
- 检查项目根目录是否存在 .env.example。
- 输出 PASS 或 FAIL 状态。
- 检查失败时返回非零退出码。

本工具仅进行基础检查，不会安装依赖、下载模型或访问外部 API。

## 2. 使用方法

在项目根目录打开 PowerShell，执行：

```powershell
py scripts/check_environment.py
```

也可以指定已安装的 Python 版本：

```powershell
py -3.12 scripts/check_environment.py
```

## 3. 输出示例

使用 Python 3.12 时，可能得到：

```text
ResearchPilot-RAG environment preflight
[FAIL] Python 3.12 detected; documented version: 3.13
[PASS] requirements.txt exists
[PASS] .env.example exists
Please fix failed checks before installation.
```

由于项目当前文档要求 Python 3.13，因此上述版本检查失败属于预期行为。

## 4. 自动化测试

运行：

```powershell
py -3.12 -m unittest discover -s tests -p "test_environment_preflight.py" -v
```

测试覆盖：

- 符合要求的模拟环境
- 不符合要求的 Python 版本
- 缺少必要项目文件的情况

## 5. 使用限制

- 本工具仅进行基础环境检查，不验证全部第三方依赖。
- 不检查本地模型、GPU、网络连接或 DeepSeek API Key。
- Python 3.13 的通过场景已通过单元测试模拟，但尚需在真实 Python 3.13 环境下验证。
- 通过本检查不代表完整 RAG 系统一定能够正常运行。
