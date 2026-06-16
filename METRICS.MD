# Why We Track What We Track: Our Metrics Philosophy


## 🖥️ Server Metrics: Keeping the Front Door Open

### 1. Total HTTP Requests
* **What it is:** A live heartbeat counting every single interaction hitting our API, sorted by where it’s going and what it’s doing.
* **Why it matters to us:** This is the baseline pulse of our app. If this number suddenly flatlines, we know instantly that something is broken and users can't reach us. It also tells us *when* people need us most. Are they opening up at 2 AM? Knowing these traffic patterns ensures we are always awake and scaled up when things get heavy.

### 2. Failed HTTP Requests
* **What it is:** A dedicated tracker for server-side crashes and system failures (`5xx` errors).
* **Why it matters to us:** On a standard website, an error page is an annoyance. On a mental health chatbot, a failed request is a broken promise—a dropped connection with someone who might be in distress. We isolate server errors so we can spot immediately if a bad deployment broke our text generation or if our databases (`Qdrant`/`Redis`) are lagging, allowing us to fix it before users notice.

### 3. Response Over Time (Latency)
* **What it is:** A continuous look at exactly how many seconds it takes for our bot to reply to a user message.
* **Why it matters to us:** Empathy requires presence, and a lagging bot feels detached and uncaring. If our AI or data retrieval takes longer than 10 seconds, a user might think the app is abandoned and log off. By tracking `p95` and `p99` response times, we catch slow-downs early, ensuring the user experience always feels fluid, warm, and responsive.

---

## Model / NLP Metrics: Is the AI Being a Good Listener?

### 4. Intent Distribution
* **What it is:** A breakdown of what our users are actually trying to achieve (e.g., looking for immediate coping tools, saying hello, or asking things `out_of_scope`).
* **Why it matters to us:** This is our reality check. It tells us if the bot is actually helping with the issues it was built for. If we see a massive spike in "out of scope" requests, it means our users need help with things our system can't safely handle yet, signaling that we need to expand our dataset or refine our guidance.

### 5. RAG Pipeline Activity
* **What it is:** Monitoring whether the bot grounded its answer in our safe, vetted counseling database (`used=true`) or had to rely on its raw LLM knowledge (`used=false`).
* **Why it matters to us:** We built a custom retrieval system for a reason: to keep conversations grounded in proven, safe mental health frameworks. If the bot starts "freestyling" answers too often (`used=false`), it means our system isn't pulling from our trusted knowledge base, and we need to fix the internal logic immediately.

### 6. Emotional Trends
* **What it is:** An aggregated count of the primary emotions detected in user messages (sadness, fear, anger, relief, etc.).
* **Why it matters to us:** Knowing the emotional landscape of our community helps us tailor the app's boundaries. If fear or sadness is spiking across the board, it reminds us how gentle and supportive our bot needs to be. It's also a great sanity check for our code—if a mental health platform registers 99% "joy," we know our emotion classifier is broken.

### 7. Language Distribution
* **What it is:** A breakdown of the languages our users are typing in, identified by our language detector.
* **Why it matters to us:** Support shouldn't have a language barrier. If we see a sudden influx of users typing in Arabic, Spanish, or Hindi, it tells us where our translation pipeline is working hardest. It helps us decide where to invest in native language support rather than just relying on auto-translation.

---

## Data Metrics: Staying Sustainable & Grounded

### 8. Total Token Consumption
* **What it is:** A granular look at the computational "tokens" we use across every step—translation, figuring out user intent, and generating the final response.
* **Why it matters to us:** Running advanced AI models is expensive. If a subtle change to a prompt causes token usage to skyrocket 10x overnight, it means our context windows are flooding and we're burning through our budget. This metric is our financial early-warning system, keeping our operations sustainable.

### 9. User Feedback (Thumbs Up / Down)
* **What it is:** The direct count of when a user clicks thumbs-up (helpful) or thumbs-down (unhelpful) on a response.
* **Why it matters to us:** > **Our Ultimate Truth-Teller:** We can have perfect server uptime, blazing-fast response speeds, and beautiful intent tracking, but none of it matters if the answers aren't actually helping. A spike in negative feedback tells us exactly when the tech is failing the human. This bridges the gap between clean code and real-world impact.