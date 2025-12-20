import { Link } from "react-router-dom";
import wave from "../assets/wave.svg";
import bar_chart from "../assets/bar_chart.svg";
import record_voice from "../assets/record_voice.svg";
import { ChartArea, HatGlasses, MoveRight, TrendingUpDown } from "lucide-react";

const Diagram = () => {
  return (
    <div className="grid grid-cols-[2fr_1fr_2fr] gap-4 w-[50vw] text-center bg-white px-16 py-20 rounded-tl-4xl rounded-br-4xl">
      <div>
        <img src={record_voice} className="-z-2 max-h-xs" />
        <p className="text-fg-light text-xl">Record 30s of voice</p>
      </div>
      <div className="flex justify-center items-center">
        <MoveRight size={80} color="var(--color-primary)" />
      </div>
      <div>
        <img src={bar_chart} className="-z-2 max-h-xs" />
        <p className="text-fg-light text-xl">
          Get diabetes risk score within seconds
        </p>
      </div>
    </div>
  );
};

const InfoCard = ({ icon, title, description }) => {
  return (
    <div className="p-16 rounded-xl bg-white m-8 text-center flex flex-col justify-center items-center">
      <div>{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
};

const SplashNav = () => {
  return (
    <nav className="h-24 flex flex-row justify-between items-center px-16 absolute w-full bg-[#efffff00] backdrop-blur-2xl z-2">
      <Link className="font-bold text-3xl text-primary" to="/">
        DiaVoc
      </Link>
      <div>
        <Link
          className="px-6 py-3 bg-primary text-fg-dark border-2 border-primary hover:text-primary box-border hover:bg-transparent rounded-md ml-4"
          to="/login"
        >
          Log In
        </Link>
        <Link
          className="px-6 py-3 text-primary  border-2 border-primary bg-transparent hover:bg-primary hover:text-fg-dark box-border rounded-md ml-4"
          to="/signup"
        >
          Sign Up
        </Link>
      </div>
    </nav>
  );
};

const Splash = () => {
  return (
    <>
      <SplashNav />
      <div className="h-[200vh]">
        {/* <section className="px-16 h-screen flex items-center justify-start bg-linear-to-br to-primary/30 from-primary/0"> */}
        <section className="px-16 h-screen flex items-center justify-start bg-[url(/assets/splash_bg.svg)] bg-cover">
          <div className="lg:ml-20">
            <div className="mb-8">
              <h1 className="text-7xl mb-4 font-semibold">Detect diabetes</h1>
              <h1 className="text-7xl mb-8 text-primary">with your voice</h1>
              <p className="text-2xl text-fg-light">
                Your Voice, Your Health, Your Privacy
              </p>
            </div>
          </div>
          {/* <div className="w-screen"> */}
          {/*   <img src={splash_bg} className="absolute right-0 top-0 w-screen h-screen" /> */}
          {/* </div> */}
        </section>
        <section className="px-16 h-screen flex items-center bg-primary">
          <div className="flex flex-row lg:grid lg:grid-cols-3 lg: h-[30vh]">
            <InfoCard
              icon=<HatGlasses />
              title="Privacy-First Screening"
              description="Your voice data never leaves your device—complete on-device processing ensures total privacy."
            />
            <InfoCard
              icon=<TrendingUpDown />
              title="Explainable Predictions"
              description="Understand why you received your result with transparent, interpretable AI explanations."
            />
            <InfoCard 
              icon=<ChartArea />
              title="Risk Score Prediction"
              description="Receive a personalized diabetes risk score (0-100) that quantifies your likelihood of Type-2 Diabetes based on voice biomarkers and health profile."
            />
          </div>
        </section>
      </div>
    </>
  );
};

export default Splash;
