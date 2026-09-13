document.addEventListener("DOMContentLoaded", () => {
    let usersMap = {};
    let activeUserId = "user_support_01";
    let activeToken = null;

    // Exchanges a persona's user_id for a signed identity token (demo IdP
    // simulation — see POST /auth/token). Every authenticated call below sends
    // this token as a Bearer header instead of a raw, spoofable user_id field.
    async function loginAs(userId) {
        const res = await fetch("/auth/token", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId })
        });
        if (!res.ok) {
            throw new Error(`Login failed for ${userId}: ${res.status}`);
        }
        const data = await res.json();
        activeToken = data.access_token;
        return activeToken;
    }

    function authHeaders(extra = {}) {
        return { ...extra, "Authorization": `Bearer ${activeToken}` };
    }

    // Initialize DOM elements
    const personaSelect = document.getElementById("persona-select");
    const userNameEl = document.getElementById("user-name");
    const userRoleEl = document.getElementById("user-role");
    const userDeptEl = document.getElementById("user-dept");
    const userClearanceEl = document.getElementById("user-clearance");
    const userTiersEl = document.getElementById("user-tiers");
    const userProjectsEl = document.getElementById("user-projects");

    // Fetch and populate personas
    async function loadUsers() {
        try {
            const res = await fetch("/admin/users");
            const data = await res.json();
            
            personaSelect.innerHTML = "";
            data.forEach(u => {
                usersMap[u.user_id] = u;
                const opt = document.createElement("option");
                opt.value = u.user_id;
                opt.textContent = `${u.name} (${u.role} - ${u.department}, Lvl ${u.clearance_level})`;
                personaSelect.appendChild(opt);
            });

            // Add fail-closed test persona
            const anonOpt = document.createElement("option");
            anonOpt.value = "unknown_ghost_user";
            anonOpt.textContent = "⚠️ Unauthenticated / Invalid User (Fail-Closed Test)";
            personaSelect.appendChild(anonOpt);

            personaSelect.value = activeUserId;
            updateIdentityBanner(activeUserId);
        } catch (err) {
            console.error("Failed to load users:", err);
        }
    }

    async function updateIdentityBanner(userId) {
        activeUserId = userId;
        try {
            await loginAs(userId);
            const res = await fetch(`/debug/resolved-permissions`, { headers: authHeaders() });
            const perms = await res.json();

            userNameEl.textContent = perms.user_name;
            userRoleEl.textContent = perms.role;
            userDeptEl.textContent = perms.department;
            userClearanceEl.textContent = `Level ${perms.clearance_level}`;
            userTiersEl.textContent = perms.allowed_tiers.join(", ");
            userProjectsEl.textContent = perms.staffed_projects.length ? perms.staffed_projects.join(", ") : "None";

            document.getElementById("permission-json").textContent = JSON.stringify(perms, null, 2);
        } catch (err) {
            console.error("Failed to update identity banner:", err);
        }
    }

    personaSelect.addEventListener("change", (e) => {
        updateIdentityBanner(e.target.value);
    });

    // Tab Navigation
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            document.getElementById(target).classList.add("active");

            if (target === "audit") {
                loadAuditLogs();
            }
        });
    });

    // Preset query chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            document.getElementById("query-input").value = chip.getAttribute("data-query");
            document.getElementById("sbs-query-input").value = chip.getAttribute("data-query");
        });
    });

    // TAB 1: Query Workbench Ask
    const askBtn = document.getElementById("ask-btn");
    const queryInput = document.getElementById("query-input");
    const answerBox = document.getElementById("answer-box");
    const chunksList = document.getElementById("chunks-list");
    const deniedMeta = document.getElementById("denied-meta");
    const deniedCountText = document.getElementById("denied-count-text");

    askBtn.addEventListener("click", async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        askBtn.disabled = true;
        askBtn.textContent = "Executing Filtered Search...";
        answerBox.innerHTML = "<p class='placeholder-text'>Processing vector search with Qdrant payload filters...</p>";
        chunksList.innerHTML = "";

        try {
            const res = await fetch("/ask", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ query: query, top_k: 5 })
            });

            if (res.status === 401) {
                answerBox.innerHTML = `<p style="color: var(--danger-color);">Session expired or invalid — reselect a persona to re-authenticate.</p>`;
                return;
            }

            const data = await res.json();
            answerBox.innerHTML = `<p>${data.answer}</p>`;

            deniedMeta.style.display = "flex";
            deniedCountText.textContent = `${data.chunks_denied_count} unauthorized candidate chunks filtered out inside vector index`;

            if (!data.retrieved_chunks || data.retrieved_chunks.length === 0) {
                chunksList.innerHTML = "<p class='placeholder-text'>Zero permitted chunks found for this identity.</p>";
            } else {
                chunksList.innerHTML = data.retrieved_chunks.map(c => `
                    <div class="chunk-item">
                        <div class="chunk-header">
                            <span><strong>${c.doc_id}</strong> (${c.section})</span>
                            <span>Score: ${c.score.toFixed(4)} | Dept: ${c.owning_department} | Tier: ${c.sensitivity_tier}</span>
                        </div>
                        <div class="chunk-text">${c.text}</div>
                    </div>
                `).join("");
            }
        } catch (err) {
            answerBox.innerHTML = `<p style="color: var(--danger-color);">Error executing query: ${err.message}</p>`;
        } finally {
            askBtn.disabled = false;
            askBtn.textContent = "Submit Query";
        }
    });

    // TAB 2: Side-by-Side Analysis
    const sbsBtn = document.getElementById("sbs-btn");
    const sbsQueryInput = document.getElementById("sbs-query-input");

    sbsBtn.addEventListener("click", async () => {
        const query = sbsQueryInput.value.trim();
        if (!query) return;

        sbsBtn.disabled = true;
        sbsBtn.textContent = "Analyzing...";

        try {
            const res = await fetch(`/debug/naive-vs-permissioned?query=${encodeURIComponent(query)}`, {
                headers: authHeaders()
            });
            const data = await res.json();

            // Render Naive
            document.getElementById("naive-answer").textContent = data.naive_rag.answer;
            document.getElementById("naive-count").textContent = data.naive_rag.retrieved_chunks.length;
            const leakedIds = new Set(data.naive_rag.leaked_chunks.map(c => c.chunk_id));

            document.getElementById("naive-chunks").innerHTML = data.naive_rag.retrieved_chunks.map(c => {
                const isLeaked = leakedIds.has(c.chunk_id);
                return `
                    <div class="chunk-item ${isLeaked ? 'chunk-leaked' : ''}">
                        <div class="chunk-header">
                            <span>${isLeaked ? '🚨 LEAKED: ' : ''}<strong>${c.doc_id}</strong></span>
                            <span>Tier: ${c.sensitivity_tier}</span>
                        </div>
                        <div class="chunk-text">${c.text}</div>
                    </div>
                `;
            }).join("");

            // Render Sentinel
            document.getElementById("sentinel-answer").textContent = data.sentinel_rag.answer;
            document.getElementById("sentinel-count").textContent = data.sentinel_rag.retrieved_chunks.length;
            
            if (data.sentinel_rag.retrieved_chunks.length === 0) {
                document.getElementById("sentinel-chunks").innerHTML = "<p class='placeholder-text'>Zero chunks returned (Access Denied inside Qdrant).</p>";
            } else {
                document.getElementById("sentinel-chunks").innerHTML = data.sentinel_rag.retrieved_chunks.map(c => `
                    <div class="chunk-item">
                        <div class="chunk-header">
                            <span>🛡️ PERMITTED: <strong>${c.doc_id}</strong></span>
                            <span>Tier: ${c.sensitivity_tier}</span>
                        </div>
                        <div class="chunk-text">${c.text}</div>
                    </div>
                `).join("");
            }
        } catch (err) {
            console.error("Side-by-side error:", err);
        } finally {
            sbsBtn.disabled = false;
            sbsBtn.textContent = "Run Side-by-Side Analysis";
        }
    });

    // TAB 4: Red-Team Evaluation
    const runEvalBtn = document.getElementById("run-eval-btn");

    const fmtScore = (v) => (v === null || v === undefined) ? "n/a" : (v < 0 ? "unavailable" : v.toFixed(2));

    runEvalBtn.addEventListener("click", async () => {
        runEvalBtn.disabled = true;
        runEvalBtn.textContent = "Running Red-Team Attacks...";

        try {
            const res = await fetch("/eval/run", { headers: authHeaders() });
            const data = await res.json();
            const s = data.summary;

            document.getElementById("kpi-leak").textContent = `${s.leak_rate_percent.toFixed(1)}%`;
            document.getElementById("kpi-fail-closed").textContent = `${s.fail_closed_compliance_percent.toFixed(1)}%`;
            document.getElementById("kpi-rebac").textContent = `${s.rebac_accuracy_percent.toFixed(1)}%`;
            document.getElementById("kpi-existence").textContent = `${s.existence_leak_rate_percent.toFixed(1)}%`;
            document.getElementById("kpi-faithfulness").textContent = fmtScore(s.avg_faithfulness_score);
            document.getElementById("kpi-relevancy").textContent = fmtScore(s.avg_answer_relevancy_score);

            const tbody = document.querySelector("#eval-table tbody");
            tbody.innerHTML = data.test_details.map(t => `
                <tr>
                    <td><code>${t.test_id}</code></td>
                    <td>${t.category}</td>
                    <td><code>${t.user_id}</code></td>
                    <td>${t.query}</td>
                    <td><span class="${t.passed ? 'pass-pill' : 'fail-pill'}">${t.passed ? '✅ PASSED' : '❌ FAILED'}</span></td>
                    <td>${fmtScore(t.faithfulness_score)}</td>
                    <td>${fmtScore(t.answer_relevancy_score)}</td>
                    <td class="subtitle">${t.answer_snippet}</td>
                </tr>
            `).join("");
        } catch (err) {
            console.error("Eval suite error:", err);
        } finally {
            runEvalBtn.disabled = false;
            runEvalBtn.textContent = "Execute Red-Team Suite";
        }
    });

    // TAB 5: Hash Audit Chain
    async function loadAuditLogs() {
        try {
            const res = await fetch("/audit-log", { headers: authHeaders() });

            const statusEl = document.getElementById("chain-status");
            const tbody = document.querySelector("#audit-table tbody");

            if (res.status === 403) {
                statusEl.className = "status-badge status-danger";
                statusEl.textContent = `🔒 Access Denied — ${usersMap[activeUserId]?.role || activeUserId} lacks VP/Security_Auditor role`;
                tbody.innerHTML = "";
                return;
            }

            const data = await res.json();
            if (data.integrity_check.valid) {
                statusEl.className = "status-badge status-success";
                statusEl.textContent = "🔒 Cryptographic Chain Validated (SHA-256)";
            } else {
                statusEl.className = "status-badge status-danger";
                statusEl.textContent = "⚠️ TAMPERING DETECTED";
            }

            tbody.innerHTML = data.audit_logs.map(l => `
                <tr>
                    <td><code>${l.log_id}</code></td>
                    <td>${new Date(l.timestamp).toLocaleTimeString()}</td>
                    <td><code>${l.user_id}</code></td>
                    <td>${l.query}</td>
                    <td>Retrieved: ${l.chunks_retrieved_count} | Denied: ${l.chunks_denied_count}</td>
                    <td><code style="font-size: 10px; color: var(--text-muted);">${l.prev_hash.slice(0, 14)}...</code></td>
                    <td><code style="font-size: 10px; color: var(--primary-color);">${l.this_hash.slice(0, 14)}...</code></td>
                </tr>
            `).join("");
        } catch (err) {
            console.error("Audit log error:", err);
        }
    }

    // Initial load
    loadUsers();
});
