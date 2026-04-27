// VS Code extension entry. Talks to the local FastAPI backend over HTTP/SSE.
import * as vscode from "vscode";

const cfg = () => vscode.workspace.getConfiguration("codebaseCopilot");
const serverUrl = (): string => cfg().get<string>("serverUrl") ?? "http://127.0.0.1:8000";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${serverUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return (await res.json()) as T;
}

function workspaceRoot(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel("Codebase Copilot");
  context.subscriptions.push(output);

  context.subscriptions.push(
    vscode.commands.registerCommand("codebaseCopilot.indexRepo", async () => {
      const root = workspaceRoot();
      if (!root) {
        vscode.window.showErrorMessage("Open a workspace first.");
        return;
      }
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Indexing repository…" },
        async () => {
          try {
            const result = await postJson<{ files_indexed: number; chunks_embedded: number }>(
              "/api/index",
              { repo_path: root }
            );
            vscode.window.showInformationMessage(
              `Indexed ${result.files_indexed} files, ${result.chunks_embedded} chunks.`
            );
          } catch (e) {
            vscode.window.showErrorMessage(`Index failed: ${(e as Error).message}`);
          }
        }
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codebaseCopilot.askQuestion", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Ask Codebase Copilot",
        placeHolder: "e.g. Where is authentication handled?",
      });
      if (!query) return;
      await runChat(query, output);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codebaseCopilot.findDeadCode", async () => {
      await runChat("Find dead code in this repo and explain why each is unused.", output);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codebaseCopilot.suggestTests", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const selection = editor.document.getText(editor.selection).trim();
      const word = selection || editor.document.getText(
        editor.document.getWordRangeAtPosition(editor.selection.active) ?? new vscode.Range(0, 0, 0, 0)
      );
      if (!word) {
        vscode.window.showWarningMessage("Place cursor on a symbol first.");
        return;
      }
      await runChat(`Suggest unit tests for ${word}.`, output);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codebaseCopilot.openChat", () => {
      ChatPanel.createOrShow(context.extensionUri);
    })
  );
}

async function runChat(query: string, output: vscode.OutputChannel) {
  output.show(true);
  output.appendLine(`> ${query}`);
  try {
    const res = await postJson<{ answer: string; elapsed_seconds: number }>(
      "/api/chat",
      { query, stream: false }
    );
    output.appendLine(res.answer);
    output.appendLine(`(${res.elapsed_seconds}s)`);
  } catch (e) {
    output.appendLine(`ERROR: ${(e as Error).message}`);
  }
}

/** Minimal chat webview for richer UX than the output channel. */
class ChatPanel {
  static current: ChatPanel | undefined;
  private readonly panel: vscode.WebviewPanel;

  static createOrShow(extensionUri: vscode.Uri) {
    if (ChatPanel.current) {
      ChatPanel.current.panel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "codebaseCopilotChat",
      "Codebase Copilot",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    ChatPanel.current = new ChatPanel(panel);
  }

  private constructor(panel: vscode.WebviewPanel) {
    this.panel = panel;
    this.panel.webview.html = this.html();
    this.panel.onDidDispose(() => (ChatPanel.current = undefined));
    this.panel.webview.onDidReceiveMessage(async (msg) => {
      if (msg.type === "ask") {
        try {
          const res = await postJson<{ answer: string }>("/api/chat", { query: msg.query, stream: false });
          this.panel.webview.postMessage({ type: "answer", text: res.answer });
        } catch (e) {
          this.panel.webview.postMessage({ type: "answer", text: `Error: ${(e as Error).message}` });
        }
      }
    });
  }

  private html(): string {
    return /* html */ `<!DOCTYPE html>
<html><head><meta charset="UTF-8" /><style>
  body { font-family: var(--vscode-font-family); padding: 12px; }
  #log { white-space: pre-wrap; margin-bottom: 12px; }
  .user { color: var(--vscode-textLink-foreground); }
  .assistant { color: var(--vscode-foreground); margin: 8px 0; }
  textarea { width: 100%; box-sizing: border-box; min-height: 60px; }
  button { padding: 6px 12px; margin-top: 4px; }
</style></head><body>
<div id="log"></div>
<textarea id="q" placeholder="Ask about your code…"></textarea>
<button id="send">Send</button>
<script>
  const vscode = acquireVsCodeApi();
  const log = document.getElementById('log');
  const q = document.getElementById('q');
  document.getElementById('send').onclick = () => {
    const query = q.value.trim();
    if (!query) return;
    log.innerHTML += '<div class="user">> ' + escape(query) + '</div>';
    q.value = '';
    vscode.postMessage({ type: 'ask', query });
  };
  window.addEventListener('message', e => {
    if (e.data.type === 'answer') {
      log.innerHTML += '<div class="assistant">' + escape(e.data.text) + '</div>';
    }
  });
  function escape(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]);}
</script>
</body></html>`;
  }
}

export function deactivate() {}