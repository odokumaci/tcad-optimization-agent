"""Test MCP tool discovery and calls against a live surrogate API."""

import asyncio
import json
import os
import subprocess
import sys

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from smoke_test_api_server import BASE_URL, wait_until_ready


DESIGN = {
    "gate_length_nm": 50.0,
    "oxide_thickness_nm": 1.2,
    "halo_peak_doping_1e19": 3.0,
    "junction_depth_nm": 30.0,
}
EXPECTED_TOOLS = {
    "check_api_health",
    "get_valid_design_ranges",
    "predict_device_metrics",
    "predict_idvg_curves",
    "optimize_device_design",
    "verify_design_with_tcad",
}


async def test_mcp_protocol() -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["tcad_mcp_server.py"],
        env={**os.environ, "SURROGATE_API_URL": BASE_URL},
        cwd=os.getcwd(),
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            listed = await session.list_tools()
            tool_names = {tool.name for tool in listed.tools}
            if tool_names != EXPECTED_TOOLS:
                raise AssertionError(
                    f"Unexpected MCP tools: {sorted(tool_names)}"
                )

            for name, arguments in (
                ("check_api_health", {}),
                ("get_valid_design_ranges", {}),
                ("predict_device_metrics", DESIGN),
                ("predict_idvg_curves", DESIGN),
                (
                    "optimize_device_design",
                    {
                        "max_ioff_ua_per_um": 0.001,
                        "max_ss_mv_per_dec": 85.0,
                        "max_dibl_mv_per_v": 50.0,
                        "samples": 256,
                        "top_k": 3,
                    },
                ),
            ):
                result = await session.call_tool(name, arguments=arguments)
                if result.isError:
                    raise AssertionError(f"MCP tool {name} failed: {result.content}")
                if result.structuredContent is None:
                    raise AssertionError(
                        f"MCP tool {name} returned no structured content"
                    )
                if name == "predict_device_metrics":
                    print(json.dumps(result.structuredContent, sort_keys=True))
                print(f"{name}=passed")


def main() -> None:
    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "surrogate_api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ]
    )
    try:
        with httpx.Client(timeout=10.0) as client:
            wait_until_ready(client, api_process)
        asyncio.run(test_mcp_protocol())
    finally:
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()
            api_process.wait(timeout=5)


if __name__ == "__main__":
    main()
