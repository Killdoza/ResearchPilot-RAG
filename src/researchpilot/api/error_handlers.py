"""统一、不泄漏内部栈和密钥的 API 错误响应。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    code: str
    message: str


def error_response(*, request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", "unknown")
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "trace_id": trace_id}},
        headers={"X-Trace-Id": trace_id},
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(request=request, status_code=exc.status_code, code=exc.code, message=exc.message)


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return error_response(request=request, status_code=422, code="invalid_request", message=str(exc))


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        request=request,
        status_code=500,
        code="internal_error",
        message="服务内部错误；请提供 X-Trace-Id 以便排查。",
    )
