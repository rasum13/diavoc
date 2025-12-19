import { useEffect, useRef, useState } from "react";

const useAudioRecorder = () => {
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
 const chunksRef = useRef([]);

  const timerRef = useRef(null);

  const [seconds, setSeconds] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recording, setRecording] = useState(false);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
    });

    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
      // TODO: Send audio
      // sendAudio(blob)
    };

    setSeconds(0);
    mediaRecorder.start();
    mediaRecorderRef.current = mediaRecorder;
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }

    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    setRecording(false);
  };

  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }

    return () => clearInterval(timerRef.current);
  }, [recording]);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  return {
    recording,
    seconds,
    audioBlob,
    startRecording,
    stopRecording,
  };
};

export default useAudioRecorder;
