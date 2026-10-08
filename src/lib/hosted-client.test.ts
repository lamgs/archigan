import { afterEach, describe, expect, it, vi } from "vitest";
import { cancelHostedTask, createHostedTask, downloadHostedModel, fetchHostedTask } from "./hosted-client";

afterEach(() => vi.unstubAllGlobals());

describe("hosted client is provider-aware", () => {
  it.each(["meshy", "tripo", "hunyuan3d-rapid", "hunyuan3d-pro"] as const)("sends %s in the body and the ?provider= query", async (provider) => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ taskId: "t1", task: { providerTaskId: "t1", status: "queued" } }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await createHostedTask({ prompt: "p", refinement: "", code: "c", provider });
    expect(JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)).toMatchObject({ provider, confirmSpend: true });
    await fetchHostedTask(provider, "t 1", "c");
    await cancelHostedTask(provider, "t 1", "c");
    await downloadHostedModel(provider, "t 1", "c");
    const urls = fetchMock.mock.calls.slice(1).map((call) => (call as unknown as [string])[0]);
    expect(urls).toEqual([`/api/generate/t%201?provider=${provider}`, `/api/generate/t%201?provider=${provider}`, `/api/generate/t%201/model?provider=${provider}`]);
  });
});
