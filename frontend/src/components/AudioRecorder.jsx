import Button from "./Button";
import useAudioRecorder from "../hooks/useAudioRecorder";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGear, faMicrophone } from "@fortawesome/free-solid-svg-icons";
import { useState } from "react";
import { api } from "../lib/axios";

const requiredAudioLength = 30;

const AudioRecorder = () => {
  const [sending, setSending] = useState(false);
  const [response, setResponse] = useState(null);
  const { recording, seconds, audioBlob, startRecording, stopRecording } =
    useAudioRecorder();

  // TODO: function to send audio data
  const sendAudio = async (audioBlob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "voice.webm");

    setSending(true);
    const res = await api.postForm("/analyze", formData);
    setResponse("Done");
    setSending(false);
  };

  return (
    <div className="text-center">
      <h2
        className={`text-center text-8xl font-medium m-4 ${recording || seconds > requiredAudioLength ? "text-primary" : "text-neutral-200"}`}
      >
        {seconds}s
      </h2>
      <div className="grid grid-cols-2 gap-4">
        {recording ? (
          <Button className="bg-red" onClick={stopRecording}>
            <div className={recording && "animate-pulse"}>
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
      {response && <p className="text-primary">{response}</p>}
    </div>
  );
};

export default AudioRecorder;
