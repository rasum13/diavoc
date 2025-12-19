import { TriangleAlert } from "lucide-react"

const Disclaimer = ({ title, message }) => {
  return (
    <div className="bg-yellow-light rounded-md p-4 border-2 border-yellow text-yellow max-w-200">
      <h3 className="text-base flex mb-2"><TriangleAlert />&nbsp;{title}</h3>
      <p className="text-sm">{message}</p>
    </div>
  )
}

export default Disclaimer
