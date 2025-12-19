import { useState } from "react";
import AudioRecorder from "../components/AudioRecorder";
import AudioUploader from "../components/AudioUploader";

const Record = () => {
  const [upload, setUpload] = useState(false);

  return (
    <div className="flex flex-col items-center justify-center w-full h-[80vh]">
      {upload ? (
        <>
          <h1 className="mt-6">Upload Audio</h1>
          <AudioUploader />
        </>
      ) : (
        <>
          <h1>Record</h1>
          <p className="mb-4 text-center">
            Record at least{" "}
            <span className="text-primary italic">30 seconds</span> of voice to
            analyze.
          </p>
          <AudioRecorder />
        </>
      )}
      <div className="m-4 text-primary hover:text-primary-dark cursor-pointer underline" onClick={() => setUpload(!upload)}>{upload ? "Record Instead" : "Upload Instead"}</div>
    </div>
  );
};

export default Record;
