// The landing page. Editorial, warm, honest — every claim on this page is
// true of the running product. Marketing voice per DESIGN.md: never say
// RAG, vectors, or any of the machinery.

import { Link } from '../router'

export function Landing() {
  return (
    <div className="mkt">
      <nav className="mkt-nav">
        <span className="wordmark">AIOS</span>
        <div className="mkt-nav-links">
          <Link to="/thesis" className="mkt-navlink">
            Thesis
          </Link>
          <Link to="/app" className="btn btn-ink">
            Open the app
          </Link>
        </div>
      </nav>

      <header className="hero">
        <p className="eyebrow reveal">A memory operating system for AI</p>
        <h1 className="reveal d1">
          Never explain
          <br />
          yourself twice.
        </h1>
        <p className="hero-sub reveal d2">
          AIOS is one continuous relationship with an AI that actually remembers — your
          projects, your decisions, your life. Kill the app. Switch the model. Come back
          in a month. It remembers.
        </p>
        <div className="hero-cta reveal d3">
          <Link to="/app" className="btn btn-ember">
            Open the app
          </Link>
          <Link to="/thesis" className="btn btn-ghost">
            Read the thesis →
          </Link>
        </div>

        <div className="demo reveal d4" aria-hidden="true">
          <div className="demo-head">
            <span className="demo-dot" />
            <span className="demo-dot" />
            <span className="demo-dot" />
            <span className="demo-title">one timeline · no sessions</span>
          </div>
          <div className="demo-body">
            <div className="demo-row user da1">
              <span className="demo-bubble user">I booked flights to Lisbon for the first week of September.</span>
            </div>
            <div className="demo-chip da2">◆ remembered — Lisbon, first week of September</div>
            <div className="demo-divider da3">
              <span>model swapped · Claude → local llama · memory intact</span>
            </div>
            <div className="demo-row user da4">
              <span className="demo-bubble user">how long until my trip?</span>
            </div>
            <div className="demo-row da5">
              <span className="demo-bubble ai">
                Six weeks — you fly to Lisbon the first week of September.
                <span className="demo-meta">◆ recalled from memory · 47ms</span>
              </span>
            </div>
          </div>
        </div>
      </header>

      <section className="band">
        <div className="feature-grid">
          <div className="feature">
            <div className="feature-glyph">◈</div>
            <h3>One timeline</h3>
            <p>
              No chat list, no “new conversation”, no starting over. Your history is a
              single continuous thread you scroll like messages — and search like a mind.
            </p>
          </div>
          <div className="feature">
            <div className="feature-glyph">⇄</div>
            <h3>Any model</h3>
            <p>
              Hosted frontier models or ones running on your own machine — swap them
              mid-conversation. The relationship survives the swap, because the memory
              never lived in the model.
            </p>
          </div>
          <div className="feature">
            <div className="feature-glyph">◆</div>
            <h3>Yours</h3>
            <p>
              Memory lives on your machine, in files you can read. Browse every belief,
              correct it, or delete it for good. Go incognito and nothing is written at
              all.
            </p>
          </div>
        </div>
      </section>

      <section className="band stats">
        <div className="stat">
          <b>~50 ms</b>
          <span>to recall the right memories</span>
        </div>
        <div className="stat">
          <b>every exchange</b>
          <span>forms memory, quietly</span>
        </div>
        <div className="stat">
          <b>0 sessions</b>
          <span>anywhere in the product</span>
        </div>
      </section>

      <section className="band closing">
        <h2>One AI. One relationship.</h2>
        <p>The first conversation is the setup. Everything after is continuity.</p>
        <Link to="/app" className="btn btn-ember big">
          Start remembering
        </Link>
      </section>

      <footer className="mkt-footer">
        <span className="wordmark small">AIOS</span>
        <span>runs entirely on localhost · your memory never leaves your machine</span>
      </footer>
    </div>
  )
}
