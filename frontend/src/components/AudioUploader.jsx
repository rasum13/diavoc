import { useState } from "react";
import Button from "./Button";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGear } from "@fortawesome/free-solid-svg-icons";
import { api } from "../lib/axios";

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    try {
      setUploading(true);
      const res = await api.postForm("/predict", formData);

      setResponse(res.data.diagnosis);
      setError(null);
    } catch (err) {
      setError(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid grid-cols-[3fr_1fr] gap-4 p-4">
      <input
        className="border-[0.5px] border-neutral-300 hover:bg-neutral-100 cursor-pointer rounded-md p-2"
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
      ></input>
      <Button className="bg-yellow" onClick={handleUpload} disabled={uploading || !file}>
        <FontAwesomeIcon
          icon={faGear}
          className={uploading && "animate-spin"}
        />{" "}
        Analyze
      </Button>
      {/* {error && <p className="text-red">{error}</p>} */}
      {response && <p className="text-primary">{response}</p>}
    </div>
  );
};

export default AudioUploader;
