// The thesis. Long-form, editorial, serif. Why this exists.

import { Link } from '../router'

export function Thesis() {
  return (
    <div className="mkt">
      <nav className="mkt-nav">
        <Link to="/" className="wordmark">
          AIOS
        </Link>
        <div className="mkt-nav-links">
          <Link to="/app" className="btn btn-ink">
            Open the app
          </Link>
        </div>
      </nav>

      <article className="thesis">
        <p className="eyebrow reveal">The thesis</p>
        <h1 className="reveal d1">
          AIOS provides persistence.
          <br />
          Models provide intelligence.
        </h1>

        <section className="reveal d2">
          <h2>Every AI you've used has amnesia</h2>
          <p>
            You've told it your name a hundred times. Your stack, your deadlines, the way
            you like your writing. Every new chat is a first date with someone you've
            known for a year. The industry's answer has been a bigger notepad per
            conversation — and a longer list of conversations in a sidebar. That's not
            memory. That's clutter with timestamps.
          </p>
        </section>

        <section className="reveal d3">
          <h2>Memory is an operating system problem</h2>
          <p>
            Your computer doesn't reload every program from scratch each time you glance
            at another window. An operating system decides what stays hot, what sleeps on
            disk, and what gets paged back the instant it's needed. AIOS treats an AI's
            attention the same way: a small kernel keeps what matters in reach, and pages
            in the right memories the moment a conversation touches them.
          </p>
          <p>
            When the model is asked about something it doesn't have in hand, it does not
            improvise. It asks its own memory for more — and the kernel answers. A page
            fault, not a hallucination.
          </p>
        </section>

        <blockquote className="pull">The LLM is stateless. The kernel owns state.</blockquote>

        <section className="reveal">
          <h2>Nothing is ever really deleted</h2>
          <p>
            Memories aren't rows to overwrite. When you change your mind, AIOS writes a
            new version on top of the old one — and keeps the history, the way you'd
            remember once having believed something else. Old conversation drifts down
            into an archive instead of vanishing. The one exception is deliberate: you can
            delete a memory outright, and that is the only way anything dies.
          </p>
        </section>

        <section className="reveal">
          <h2>There are no sessions</h2>
          <p>
            A relationship doesn't restart when you put your phone down. AIOS has one
            timeline — today at the bottom, your first hello at the top. Search doesn't
            find chats; it finds the things you said, when you said them, and what the
            assistant believes now because of them.
          </p>
        </section>

        <section className="reveal">
          <h2>Models are cattle. Memory is the pet.</h2>
          <p>
            Frontier model today, a small local one on a plane tomorrow. Because memory
            lives beneath the model, swapping one intelligence for another is a settings
            change, not a divorce. Continuity is by construction, not by promise.
          </p>
        </section>

        <section className="reveal">
          <h2>Private by construction</h2>
          <p>
            The memory lives on your machine, in files you can open. Incognito mode talks
            without writing; paused mode remembers without learning. These are enforced
            in the daemon that owns the memory — not by the interface promising to look
            away.
          </p>
        </section>

        <div className="thesis-cta reveal">
          <Link to="/app" className="btn btn-ember big">
            Open the app
          </Link>
        </div>
      </article>

      <footer className="mkt-footer">
        <Link to="/" className="wordmark small">
          AIOS
        </Link>
        <span>runs entirely on localhost · your memory never leaves your machine</span>
      </footer>
    </div>
  )
}
