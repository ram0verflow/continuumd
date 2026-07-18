import { RouterProvider, useRouter } from './router'
import { Landing } from './pages/Landing'
import { Thesis } from './pages/Thesis'
import App from './App'

function Pages() {
  const { path } = useRouter()
  if (path.startsWith('/app')) return <App />
  if (path.startsWith('/thesis')) return <Thesis />
  return <Landing />
}

export default function Root() {
  return (
    <RouterProvider>
      <Pages />
    </RouterProvider>
  )
}
