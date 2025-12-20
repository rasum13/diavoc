import { TriangleAlert } from "lucide-react"

const Disclaimer = ({ title, message, className }) => {
  return (
    <div className={`bg-yellow-light rounded-md p-4 border border-yellow text-yellow absolute bottom-28 lg:bottom-4 right-4 left-4 lg:left-80 ${className}`}>
      <h3 className="text-base flex mb-2"><TriangleAlert />&nbsp;{title}</h3>
      <p className="text-sm">{message}</p>
    </div>
  )
}

export default Disclaimer
