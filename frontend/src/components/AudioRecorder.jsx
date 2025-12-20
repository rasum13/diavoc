import Button from "./Button";
import useAudioRecorder from "../hooks/useAudioRecorder";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGear, faMicrophone } from "@fortawesome/free-solid-svg-icons";
import { historyItemCreateSchema } from "../schemas/history";
import { useState, useEffect } from "react";
import { api } from "../lib/axios";
import { Mic } from "lucide-react";
import Diagnosis from "./Diagnosis";

const requiredAudioLength = 30;

const AudioRecorder = () => {
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [errors, setErrors] = useState({});
  const { recording, seconds, audioBlob, startRecording, stopRecording } =
    useAudioRecorder();

  // TODO: function to send audio data
  const sendAudio = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "voice.webm");

    try {
      setSending(true);
      const res = await api.postForm("/predict", formData);

      const data = res.data;
      setResponse(data);

      setError(null);
    } catch (err) {
      setError(err);
    } finally {
      setSending(false);
    }
  };

  useEffect(() => {
    if (seconds >= requiredAudioLength) {
      stopRecording();
    }
  }, [seconds]);

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
      <div className="text-center">
        <div
          className={`flex flex-col justify-center items-center text-center text-3xl font-medium m-4 ${recording || seconds > requiredAudioLength ? "text-primary" : "text-neutral-200"}`}
        >
          <Mic
            size={160}
            className={`${recording ? "bg-primary animate-recording" : "bg-neutral-200"} rounded-full p-8 mb-4 text-fg-dark`}
          />
          {seconds}s
        </div>
        <div className="grid grid-cols-2 gap-4">
          {recording ? (
            <Button className="bg-red" onClick={stopRecording}>
              <div>
                <FontAwesomeIcon icon={faMicrophone} /> Stop
              </div>
            </Button>
          ) : (
            <Button onClick={startRecording}>
              <FontAwesomeIcon icon={faMicrophone} /> Record
            </Button>
          )}
          <Button
            className="bg-yellow"
            onClick={() => {
              sendAudio(audioBlob);
            }}
            disabled={seconds < requiredAudioLength || recording}
          >
            <FontAwesomeIcon
              icon={faGear}
              className={sending && "animate-spin"}
            />{" "}
            Analyze
          </Button>
        </div>
      </div>
      {response && <Diagnosis score={response.probability} />}
    </>
  );
};

export default AudioRecorder;
