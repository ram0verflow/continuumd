// A ten-line router. Three pages, no dependency.

import { createContext, useContext, useEffect, useState } from 'react'

const RouterCtx = createContext<{ path: string; go: (p: string) => void }>({
  path: '/',
  go: () => {},
})

export function useRouter() {
  return useContext(RouterCtx)
}

export function RouterProvider({ children }: { children: React.ReactNode }) {
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname)
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])
  const go = (p: string) => {
    if (p === path) return
    window.history.pushState({}, '', p)
    setPath(p)
    window.scrollTo(0, 0)
  }
  return <RouterCtx.Provider value={{ path, go }}>{children}</RouterCtx.Provider>
}

export function Link({ to, className, children }: { to: string; className?: string; children: React.ReactNode }) {
  const { go } = useRouter()
  return (
    <a
      href={to}
      className={className}
      onClick={(e) => {
        e.preventDefault()
        go(to)
      }}
    >
      {children}
    </a>
  )
}
