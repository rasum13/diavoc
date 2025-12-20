import { useState, useEffect } from "react";
import Button from "./Button";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGear } from "@fortawesome/free-solid-svg-icons";
import { historyItemCreateSchema } from "../schemas/history";
import { api } from "../lib/axios";
import Diagnosis from "./Diagnosis";

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [errors, setErrors] = useState({});
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

      setResponse(res.data);
      setError(null);
    } catch (err) {
      setError(err);
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    const addHistory = async () => {
      if (response) {
        console.log(response);
        const result = historyItemCreateSchema.safeParse({
          score: response.probability,
        });

        if (!result.success) {
          const fieldErrors = {};
          for (const [key, message] of Object.entries(
            result.error.flatten().fieldErrors,
          )) {
            fieldErrors[key] = message;
          }
          setErrors(fieldErrors);
          console.error("Reusult", fieldErrors);
          return;
        }
        //
        try {
          await api.post("/history/add", result.data);
          console.log("History added successfully");
          //     navigate("/login");
        } catch (err) {
          console.error("API Error: ", err.response?.data);
          //     const detail = err.response?.data?.detail;
          //
          //     if (Array.isArray(detail)) {
          //       detail.forEach((e) => {
          //         setErrors((prev) => ({ ...prev, [e.loc.at(-1)]: e.msg }));
          //       });
          //     } else if (typeof detail === "string") {
          //       setErrors({ _form: detail });
          //     } else {
          //       setErrors({ _form: "Unexpected error" });
          //     }
        }
      }
    };
    addHistory();
  }, [response]);

  return (
    <>
      <div className="grid grid-cols-[3fr_1fr] gap-4 p-4">
        <input
          className="border-[0.5px] border-neutral-300 hover:bg-neutral-100 cursor-pointer rounded-md p-2"
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
        ></input>
        <Button
          className="bg-yellow"
          onClick={handleUpload}
          disabled={uploading || !file}
        >
          <FontAwesomeIcon
            icon={faGear}
            className={uploading && "animate-spin"}
          />{" "}
          Analyze
        </Button>
        {/* {error && <p className="text-red">{error}</p>} */}
      </div>
      {response && <Diagnosis score={response.probability} />}
    </>
  );
};

export default AudioUploader;
