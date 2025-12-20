import { Link } from "react-router-dom";
import wave from "../assets/wave.svg";
import bar_chart from "../assets/bar_chart.svg";
import record_voice from "../assets/record_voice.svg";
import { ArrowRightCircle, ChartArea, HatGlasses, MoveRight, TrendingUpDown } from "lucide-react";

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
    <div className="p-12 rounded-2xl bg-white m-8 text-center grid grid-rows-[1fr_1fr] items-center shadow-[0_0_16px_1px] hover:shadow[0_0_24px_2px] hover:shadow-white-90 shadow-white/80">
      <div className="flex justify-center items-center rounded-2xl p-8">
        {icon}
      </div>
      <div className="flex flex-col justify-start h-full mt-4">
        <h3 className="mb-4">{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
};

const SplashNav = () => {
  return (
    <nav className="h-28 flex flex-row justify-between items-center px-16 absolute w-full bg-[#efffff42] backdrop-blur-2xl z-2">
      <Link className="font-bold text-5xl text-primary" to="/">
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
  const cardIconSize = 80;

  return (
    <>
      <SplashNav />
      <div className="h-[200vh]">
        {/* <section className="px-16 h-screen flex items-center justify-start bg-linear-to-br to-primary/30 from-primary/0"> */}
        <section className="px-16 h-screen flex items-center justify-start bg-[url(/assets/splash_bg.svg)] bg-cover">
          <div className="lg:ml-20">
            <div className="mb-8">
              <h1 className="text-7xl mb-4 font-semibold">Detect diabetes</h1>
              <h1 className="text-7xl mb-8 text-primary drop-shadow-2xl drop-shadow-primary/60">
                with your voice
              </h1>
              <p className="text-2xl text-fg-light">
                Your Voice, Your Health, Your Privacy
              </p>
            </div>
          </div>
          {/* <div className="w-screen"> */}
          {/*   <img src={splash_bg} className="absolute right-0 top-0 w-screen h-screen" /> */}
          {/* </div> */}
        </section>
        <section className="px-16 lg:h-[80vh] flex flex-col justify-center items-center bg-primary">
          <h1 className="text-5xl text-fg-dark">Why DiaVoc?</h1>
          <div>
            <div className="flex flex-col lg:flex-row lg:grid lg:grid-cols-3 lg:h-[30vh]">
              <InfoCard
                icon=<HatGlasses size={cardIconSize} />
                title="Privacy-First Screening"
                description="Your voice data never leaves your device—complete on-device processing ensures total privacy."
              />
              <InfoCard
                icon=<TrendingUpDown size={cardIconSize} />
                title="Explainable Predictions"
                description="Understand why you received your result with transparent, interpretable AI explanations."
              />
              <InfoCard
                icon=<ChartArea size={cardIconSize} />
                title="Risk Score Prediction"
                description="Receive a personalized diabetes risk score (0-100) that quantifies your likelihood of Type-2 Diabetes based on voice biomarkers and health profile."
              />
            </div>
          </div>
        </section>
        <section className="px-16 h-[40vh] flex flex-col justify-center items-center">
          <h1 className="text-5xl text-primary font-semibold">Start predicting diabetes with voice</h1>
          <Link className="flex flex-row items-center justify-center transition border-2 bg-white border-primary px-8 py-4 rounded-md m-8 text-2xl font-medium text-primary hover:text-fg-dark hover:bg-primary" to="/signup">Get Started<ArrowRightCircle className="ml-2" /></Link>
        </section>
      </div>
    </>
  );
};

export default Splash;
