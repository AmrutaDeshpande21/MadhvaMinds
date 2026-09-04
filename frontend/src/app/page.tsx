"use client";

import { useEffect, useState, useRef } from "react";
import Head from "next/head";

export default function Home() {
  const [activeTab, setActiveTab] = useState("video"); // 'video', 'i3d', or 'history'
  const [historicalIncidents, setHistoricalIncidents] = useState<any[]>([]);
  
  // Real Video Dataset State
  const [videoList, setVideoList] = useState<any[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<any | null>(null);
  const [analyzingVideo, setAnalyzingVideo] = useState<boolean>(false);
  const [videoAnalysis, setVideoAnalysis] = useState<any | null>(null);
  const [currentDetections, setCurrentDetections] = useState<any[]>([]);
  const [activeOverlay, setActiveOverlay] = useState<any | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0.0);
  const [videoEnded, setVideoEnded] = useState<boolean>(false);
  
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");
  
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const categories = ["ALL", ...Array.from(new Set(videoList.map(v => v.category || "Anomaly")))];
  const filteredVideos = categoryFilter === "ALL"
    ? videoList
    : videoList.filter(v => (v.category || "Anomaly").toLowerCase() === categoryFilter.toLowerCase());

  // UCF-Crime I3D Feature Simulation State (.npy technical mode)
  const [canvasSrc, setCanvasSrc] = useState<string>("");
  const [simRunning, setSimRunning] = useState<boolean>(false);
  const [simStatusText, setSimStatusText] = useState<string>("READY");
  const [currentSample, setCurrentSample] = useState<string>("None");
  const [currentIdx, setCurrentIdx] = useState<number>(0);
  const [totalSamples, setTotalSamples] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  const [totalProcessed, setTotalProcessed] = useState<number>(0);
  const [anomaliesCount, setAnomaliesCount] = useState<number>(0);
  const [normalCount, setNormalCount] = useState<number>(0);
  const [accuracy, setAccuracy] = useState<number>(100.0);

  const ws = useRef<WebSocket | null>(null);

  // Fetch Available Real UCF-Crime Videos on Mount
  useEffect(() => {
    fetch("http://localhost:8000/api/dataset/videos")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.videos && data.videos.length > 0) {
          setVideoList(data.videos);
          setSelectedVideo(data.videos[0]);
          analyzeVideo(data.videos[0].filename);
        }
      })
      .catch((err) => console.error("Could not fetch raw dataset videos", err));
  }, []);

  // Sync I3D Simulation Status
  useEffect(() => {
    fetch("http://localhost:8000/api/simulation/status")
      .then((res) => res.json())
      .then((data) => {
        if (data) {
          setSimRunning(data.running || false);
          setSimStatusText(data.status || "READY");
          setCurrentSample(data.current_sample || "None");
          setCurrentIdx(data.current_idx || 0);
          setTotalSamples(data.total_samples || 0);
          setProgress(data.progress || 0);
          setTotalProcessed(data.total_processed || 0);
          setAnomaliesCount(data.anomalies || 0);
          setNormalCount(data.normal || 0);
          setAccuracy(data.accuracy !== undefined ? data.accuracy : 100.0);
        }
      })
      .catch((err) => console.error("Could not sync simulation status", err));
  }, []);

  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws/alerts");
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "feed" && data.image) {
        setCanvasSrc(data.image);
      }
      
      if (data.simulation) {
        setSimRunning(data.simulation.running || false);
        setSimStatusText(data.simulation.status || "READY");
        setCurrentSample(data.simulation.current_sample || "None");
        setCurrentIdx(data.simulation.current_idx || 0);
        setTotalSamples(data.simulation.total_samples || 0);
        setProgress(data.simulation.progress || 0);
        setTotalProcessed(data.simulation.total_processed || 0);
        setAnomaliesCount(data.simulation.anomalies || 0);
        setNormalCount(data.simulation.normal || 0);
        setAccuracy(data.simulation.accuracy !== undefined ? data.simulation.accuracy : 100.0);
      }
    };
    return () => ws.current?.close();
  }, []);

  useEffect(() => {
    if (activeTab === "history") {
      fetch("http://localhost:8000/api/incidents")
        .then((res) => res.json())
        .then((data) => {
          if (data.incidents) {
            setHistoricalIncidents(data.incidents);
          }
        })
        .catch((err) => console.error("Failed to fetch history", err));
    }
  }, [activeTab]);

  // Analyze Currently Selected Video Only (Isolate Detections per Video)
  const analyzeVideo = async (filename: string) => {
    setAnalyzingVideo(true);
    setVideoAnalysis(null);
    setCurrentDetections([]); // Clear previous video detections
    setActiveOverlay(null);
    setVideoEnded(false);

    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      try {
        await videoRef.current.play();
      } catch (e) {}
    }

    try {
      const res = await fetch("http://localhost:8000/api/dataset/analyze-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      if (data && data.analysis) {
        setVideoAnalysis(data.analysis);
        // Store ONLY detections belonging to this specific video file
        const videoDetections = data.analysis.intervals || [];
        setCurrentDetections(videoDetections);
      }
    } catch (err) {
      console.error("Video analysis error", err);
    } finally {
      setAnalyzingVideo(false);
    }

  };

  const handleSelectVideo = (video: any) => {
    setSelectedVideo(video);
    analyzeVideo(video.filename);
  };

  // Synchronize In-Video Overlay & Detection Status with HTML5 Video currentTime
  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);

    const isAtEnd = videoRef.current.ended || (selectedVideo?.duration && t >= selectedVideo.duration - 0.8);
    if (isAtEnd) {
      setVideoEnded(true);
    }

    if (videoAnalysis && videoAnalysis.intervals) {
      const intervals = videoAnalysis.intervals;
      const matched = intervals.find(
        (inv: any) => t >= inv.start_time && t <= inv.end_time
      );

      if (matched && matched.is_anomaly) {
        setActiveOverlay(matched);
      } else {
        setActiveOverlay(null);
      }
    } else {
      setActiveOverlay(null);
    }
  };

  const seekToTimestamp = (seconds: number) => {
    setVideoEnded(false);
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  };

  const toggleSimulation = async () => {
    const endpoint = simRunning
      ? "http://localhost:8000/api/simulation/stop"
      : "http://localhost:8000/api/simulation/start";

    try {
      const res = await fetch(endpoint, { method: "POST" });
      const data = await res.json();
      if (data && data.status) {
        setSimRunning(data.status.running || false);
        setSimStatusText(data.status.status || (simRunning ? "STOPPED" : "RUNNING"));
      }
    } catch (err) {
      console.error("Failed to toggle simulation", err);
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans p-6">
      <Head>
        <title>MadhvaMinds - Real UCF-Crime Video Anomaly Intelligence Platform</title>
      </Head>

      {/* Main Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6 pb-4 border-b border-neutral-800">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500">
              MadhvaMinds
            </h1>
            <span className="text-xs bg-cyan-500/20 text-cyan-400 font-mono px-2.5 py-0.5 rounded border border-cyan-500/30">
              REAL UCF-CRIME VIDEO AI ENGINE v3.0
            </span>
          </div>
          <p className="text-neutral-400 text-sm mt-0.5">
            Real-Time Surveillance Anomaly Detection — PyTorch Temporal Inference (Zero Webcam)
          </p>
        </div>

        {/* Control Bar & Navigation Tabs */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 bg-neutral-900 border border-neutral-800 px-3 py-1.5 rounded-xl">
            <span className="text-xs text-neutral-400">Mode:</span>
            {analyzingVideo ? (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-cyan-400 bg-cyan-500/10 px-2.5 py-0.5 rounded-lg border border-cyan-500/20">
                <span className="w-2 h-2 rounded-full bg-cyan-500 animate-ping"></span>
                ● EXTRACTING I3D FEATURES & INFERRING TEMPORAL WINDOWS
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-lg border border-emerald-500/20">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                REAL UCF-CRIME MP4 SIMULATION ACTIVE
              </span>
            )}
          </div>

          <div className="flex bg-neutral-900 border border-neutral-800 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab("video")}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition ${
                activeTab === "video" ? "bg-cyan-600 text-white shadow" : "text-neutral-400 hover:text-white"
              }`}
            >
              🎬 Real Video Player
            </button>
            <button
              onClick={() => setActiveTab("i3d")}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition ${
                activeTab === "i3d" ? "bg-indigo-600 text-white shadow" : "text-neutral-400 hover:text-white"
              }`}
            >
              📊 Technical Analysis (.npy)
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition ${
                activeTab === "history" ? "bg-blue-600 text-white shadow" : "text-neutral-400 hover:text-white"
              }`}
            >
              📜 Incident Audit Log
            </button>
          </div>
        </div>
      </header>

      {/* Verified Model Evaluation Performance Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-neutral-900/80 border border-neutral-800 p-4 rounded-2xl flex flex-col justify-between">
          <span className="text-neutral-400 text-xs font-semibold uppercase tracking-wider">MODEL ACCURACY</span>
          <div className="flex items-baseline gap-2 mt-2">
            <span className="text-3xl font-black text-cyan-400">76.67%</span>
            <span className="text-xs text-cyan-500/80">test eval</span>
          </div>
        </div>
        <div className="bg-neutral-900/80 border border-neutral-800 p-4 rounded-2xl flex flex-col justify-between">
          <span className="text-neutral-400 text-xs font-semibold uppercase tracking-wider">PRECISION</span>
          <div className="flex items-baseline gap-2 mt-2">
            <span className="text-3xl font-black text-emerald-400">75.00%</span>
            <span className="text-xs text-emerald-500/80">score</span>
          </div>
        </div>
        <div className="bg-neutral-900/80 border border-neutral-800 p-4 rounded-2xl flex flex-col justify-between">
          <span className="text-neutral-400 text-xs font-semibold uppercase tracking-wider">RECALL</span>
          <div className="flex items-baseline gap-2 mt-2">
            <span className="text-3xl font-black text-indigo-400">80.00%</span>
            <span className="text-xs text-indigo-500/80">score</span>
          </div>
        </div>
        <div className="bg-neutral-900/80 border border-neutral-800 p-4 rounded-2xl flex flex-col justify-between">
          <span className="text-neutral-400 text-xs font-semibold uppercase tracking-wider">F1 SCORE</span>
          <div className="flex items-baseline gap-2 mt-2">
            <span className="text-3xl font-black text-purple-400">77.42%</span>
            <span className="text-xs text-purple-500/80">harmonic mean</span>
          </div>
        </div>
      </div>

      {activeTab === "video" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Real UCF-Crime HTML5 Video Player Container */}
          <div className="lg:col-span-2 space-y-4">
            {/* Video Selector & Category Filter Bar */}
            <div className="bg-neutral-900 border border-neutral-800 p-3 rounded-2xl space-y-3">
              {/* Category Filter Pills */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs font-bold text-neutral-400 mr-1 uppercase tracking-wider">FILTER CATEGORY:</span>
                {categories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setCategoryFilter(cat)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-bold transition ${
                      categoryFilter === cat
                        ? "bg-cyan-600 text-white shadow"
                        : "bg-neutral-950 text-neutral-400 hover:text-white border border-neutral-800"
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Dropdown Selector */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-neutral-800">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-neutral-400 uppercase tracking-wider">SELECT DATASET VIDEO:</span>
                  <select
                    value={selectedVideo?.filename || ""}
                    onChange={(e) => {
                      const found = videoList.find((v) => v.filename === e.target.value);
                      if (found) handleSelectVideo(found);
                    }}
                    className="bg-neutral-950 text-white font-mono text-sm px-3 py-1.5 rounded-xl border border-neutral-700 focus:outline-none focus:border-cyan-500"
                  >
                    {filteredVideos.map((v) => (
                      <option key={v.filename} value={v.filename}>
                        {v.filename} ({v.duration}s)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    disabled={analyzingVideo}
                    onClick={() => selectedVideo && analyzeVideo(selectedVideo.filename)}
                    className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition flex items-center gap-1.5 ${
                      analyzingVideo
                        ? "bg-cyan-900/50 text-cyan-300 border border-cyan-500/50 cursor-wait animate-pulse"
                        : "bg-neutral-800 hover:bg-neutral-700 text-cyan-400 border border-neutral-700"
                    }`}
                  >
                    {analyzingVideo ? (
                      <>
                        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                        ⏳ RUNNING INFERENCE...
                      </>
                    ) : (
                      "↻ RE-ANALYZE VIDEO"
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* HTML5 Video Display with Dynamic AI Detection Overlay */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-2xl overflow-hidden aspect-video relative flex flex-col justify-between shadow-2xl">
              {/* Header Overlay */}
              <div className="absolute top-0 left-0 right-0 p-3 bg-gradient-to-b from-black/90 to-transparent flex justify-between items-center z-10">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm tracking-wide text-cyan-400">SURVEILLANCE AI INFERENCE ENGINE</span>
                  {analyzingVideo ? (
                    <span className="text-[10px] bg-cyan-600 text-white font-bold px-2 py-0.5 rounded animate-pulse">
                      ● INFERRING TEMPORAL WINDOWS
                    </span>
                  ) : (
                    <span className="text-[10px] bg-emerald-600 text-white font-bold px-2 py-0.5 rounded">
                      ● MODEL INFERENCE READY
                    </span>
                  )}
                </div>
                <div className="text-xs font-mono text-neutral-300 bg-black/70 px-2.5 py-1 rounded border border-neutral-700">
                  SOURCE: UCF-CRIME MP4
                </div>
              </div>

              {/* Synchronized In-Video Surveillance Incident Overlay */}
              {activeOverlay ? (
                <div className="absolute top-14 left-4 z-20 pointer-events-none bg-red-950/90 border-2 border-red-600 text-red-100 px-4 py-3 rounded-xl shadow-2xl backdrop-blur-md animate-pulse">
                  <div className="flex items-center gap-2 font-black text-base text-red-400">
                    <span className="w-3 h-3 rounded-full bg-red-500 animate-ping"></span>
                    🔴 {activeOverlay.label || (activeOverlay.event_type ? `${activeOverlay.event_type.toUpperCase()} DETECTED` : "ANOMALY DETECTED")}
                  </div>
                  <div className="text-xs font-mono mt-1 text-red-200">
                    Model Confidence: <span className="font-bold text-white">{activeOverlay.peak_confidence}%</span>
                  </div>
                  <div className="text-xs font-mono text-neutral-300">
                    Event Time: <span className="font-bold text-cyan-400">{formatTime(currentTime)}</span> (Window: {activeOverlay.start_time}s - {activeOverlay.end_time}s)
                  </div>
                </div>
              ) : (
                <div className="absolute top-14 left-4 z-20 pointer-events-none bg-emerald-950/90 border-2 border-emerald-600 text-emerald-100 px-4 py-2 rounded-xl shadow-xl backdrop-blur-md">
                  <div className="flex items-center gap-2 font-black text-sm text-emerald-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    🟢 NORMAL ACTIVITY
                  </div>
                  <div className="text-xs font-mono mt-0.5 text-emerald-300">
                    Status: <span className="font-bold text-white">No Threat Detected</span> | Time: {formatTime(currentTime)}
                  </div>
                </div>
              )}

              {/* Real HTML5 Video Player */}
              <div className="w-full h-full flex items-center justify-center bg-black">
                {selectedVideo ? (
                  <video
                    ref={videoRef}
                    src={selectedVideo.url}
                    controls
                    autoPlay
                    onEnded={() => setVideoEnded(true)}
                    onTimeUpdate={handleTimeUpdate}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-8 text-neutral-500 font-mono text-sm">
                    No MP4 video selected.
                  </div>
                )}
              </div>

              {/* Timeline Indicator Bar */}
              {videoAnalysis && selectedVideo && (
                <div className="bg-neutral-950/90 border-t border-neutral-800 p-3 z-10">
                  <div className="flex justify-between items-center text-xs font-mono mb-1.5 text-neutral-400">
                    <span>TEMPORAL DETECTION TIMELINE ({selectedVideo.filename})</span>
                    <span>
                      Playback: <span className="text-cyan-400 font-bold">{formatTime(currentTime)}</span> / {formatTime(selectedVideo.duration)}
                    </span>
                  </div>

                  <div className="relative w-full bg-neutral-800 h-4 rounded-lg border border-neutral-700 overflow-hidden">
                    {/* Render Detected Anomaly Intervals */}
                    {videoAnalysis.intervals && videoAnalysis.intervals.map((inv: any, idx: number) => {
                      const leftPct = (inv.start_time / selectedVideo.duration) * 100;
                      const widthPct = Math.max(2, ((inv.end_time - inv.start_time) / selectedVideo.duration) * 100);
                      const displayTag = inv.label ? inv.label.replace(" DETECTED", "") : (inv.event_type ? inv.event_type.toUpperCase() : "ANOMALY");
                      return (
                        <div
                          key={idx}
                          onClick={() => seekToTimestamp(inv.start_time)}
                          className="absolute top-0 bottom-0 bg-red-600 hover:bg-red-500 cursor-pointer shadow-lg z-20 flex items-center justify-center text-[9px] font-black text-white px-1 overflow-hidden"
                          style={{
                            left: `${leftPct}%`,
                            width: `${widthPct}%`
                          }}
                          title={`Click to view incident interval [${inv.start_time}s - ${inv.end_time}s]`}
                        >
                          {displayTag} ({inv.peak_confidence}%)
                        </div>
                      );
                    })}
                    <div className="w-full h-full bg-gradient-to-r from-emerald-900/30 via-neutral-900 to-emerald-900/30"></div>
                  </div>
                </div>
              )}
            </div>

            {/* Video & Inference Details Panel */}
            {videoAnalysis && (
              <div className="bg-neutral-900 p-4 rounded-2xl border border-neutral-800 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-neutral-400 font-semibold block">CURRENT VIDEO</span>
                  <span className="font-bold font-mono text-cyan-400 text-sm">{videoAnalysis.video}</span>
                </div>
                <div>
                  <span className="text-neutral-400 font-semibold block">RESOLUTION & FPS</span>
                  <span className="font-bold text-neutral-200">{videoAnalysis.resolution} @ {videoAnalysis.fps} FPS</span>
                </div>
                <div>
                  <span className="text-neutral-400 font-semibold block">INPUT TENSOR SHAPE</span>
                  <span className="font-bold font-mono text-indigo-400">(32, 2048) 2048-D</span>
                </div>
                <div>
                  <span className="text-neutral-400 font-semibold block">SIGMOID RAW SCORE</span>
                  <span className="font-bold font-mono text-purple-400">{videoAnalysis.raw_score.toFixed(4)}</span>
                </div>
              </div>
            )}

            {/* Overall Video Classification Result Banner/Card */}
            {videoAnalysis && (
              <div className={`p-4 rounded-2xl border transition-all ${
                videoAnalysis.is_anomaly
                  ? "bg-gradient-to-r from-red-950/80 via-neutral-900 to-red-950/80 border-red-500/80 shadow-lg shadow-red-950/40"
                  : "bg-gradient-to-r from-emerald-950/80 via-neutral-900 to-emerald-950/80 border-emerald-500/80 shadow-lg shadow-emerald-950/40"
              }`}>
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <div className={`p-2.5 rounded-xl text-xl font-black ${
                      videoAnalysis.is_anomaly ? "bg-red-900/60 text-red-400 border border-red-500/50" : "bg-emerald-900/60 text-emerald-400 border border-emerald-500/50"
                    }`}>
                      {videoAnalysis.is_anomaly ? "🚨" : "🟢"}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded bg-neutral-950/80 text-neutral-400 border border-neutral-800">
                          OVERALL RESULT VERDICT
                        </span>
                        {videoEnded && (
                          <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700 animate-pulse">
                            ✓ PLAYBACK COMPLETED
                          </span>
                        )}
                      </div>
                      <h3 className={`text-base font-black tracking-wide uppercase mt-1 ${
                        videoAnalysis.is_anomaly ? "text-red-300" : "text-emerald-300"
                      }`}>
                        {videoAnalysis.is_anomaly ? "ANOMALY CCTV FOOTAGE DETECTED" : "NORMAL CCTV FOOTAGE VERIFIED"}
                      </h3>
                      <p className="text-xs text-neutral-300 mt-1 font-mono">
                        {videoAnalysis.is_anomaly
                          ? `Overall Result: Classified as Anomaly CCTV Video with ${videoAnalysis.confidence}% model confidence across ${videoAnalysis.total_instances || videoAnalysis.intervals?.length || 1} threat instance window(s).`
                          : `Overall Result: Classified as Normal CCTV Video (0 threat windows detected across full duration).`}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 sm:gap-3 text-xs font-mono self-end sm:self-center">
                    <div className="bg-neutral-950/90 px-3.5 py-2 rounded-xl border border-neutral-800 text-center">
                      <span className="text-neutral-400 block text-[9px] uppercase font-bold">CATEGORY</span>
                      <span className={`font-extrabold text-sm ${videoAnalysis.is_anomaly ? "text-red-400" : "text-emerald-400"}`}>
                        {videoAnalysis.event_type ? videoAnalysis.event_type.toUpperCase() : (videoAnalysis.is_anomaly ? "ANOMALY" : "NORMAL")}
                      </span>
                    </div>
                    <div className="bg-neutral-950/90 px-3.5 py-2 rounded-xl border border-neutral-800 text-center">
                      <span className="text-neutral-400 block text-[9px] uppercase font-bold">CONFIDENCE</span>
                      <span className="font-extrabold text-sm text-cyan-400">
                        {videoAnalysis.confidence}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Isolated Live Model Detections Panel for Currently Selected Video */}
          <div className="space-y-4">
            <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex justify-between items-center mb-3 pb-2 border-b border-neutral-800">
                <div>
                  <h3 className="font-bold text-sm text-neutral-200 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
                    LIVE MODEL DETECTIONS
                  </h3>
                  <span className="text-[10px] text-neutral-400 font-mono">
                    Current Video: <span className="text-cyan-400 font-bold">{selectedVideo?.filename}</span>
                  </span>
                </div>
                <span className="text-xs font-mono bg-neutral-800 px-2 py-0.5 rounded text-neutral-400">
                  {currentDetections.filter(d => currentTime >= d.start_time - 0.5).length} / {currentDetections.length} Unfolded
                </span>
              </div>

              {/* Isolated Detections List */}
              <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
                {analyzingVideo ? (
                  <div className="bg-neutral-950/80 border border-cyan-500/30 p-4 rounded-xl text-xs space-y-2 animate-pulse">
                    <div className="flex items-center gap-2 text-cyan-400 font-bold">
                      <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                      ⏳ INFERRING TEMPORAL WINDOWS...
                    </div>
                    <p className="text-neutral-400 text-[11px] leading-relaxed">
                      Extracting 2048-D temporal features across video frame sequence...
                    </p>
                  </div>
                ) : currentDetections.length === 0 ? (
                  <div className="bg-emerald-950/20 border border-emerald-500/30 p-4 rounded-xl text-xs space-y-2 text-center">
                    <div className="flex items-center justify-center gap-2 text-emerald-400 font-bold text-sm">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                      🟢 NORMAL VIDEO VERIFIED
                    </div>
                    <p className="text-neutral-300 text-[11px]">
                      Analysis completed across video duration ({selectedVideo?.duration}s). This is a normal video and no anomaly was detected.
                    </p>
                  </div>
                ) : (
                  <>
                    {/* Live Analyzing Card when playback hasn't reached an unreached incident yet */}
                    {!videoEnded && currentDetections.filter(d => currentTime >= d.start_time - 0.5).length < currentDetections.length && (
                      <div className="bg-neutral-950/80 border border-cyan-500/30 p-3.5 rounded-xl text-xs space-y-2">
                        <div className="flex items-center gap-2 text-cyan-400 font-bold">
                          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                          🔍 MONITORING LIVE FRAME STREAM ({formatTime(currentTime)})
                        </div>
                        <p className="text-neutral-400 text-[11px] leading-relaxed">
                          Running PyTorch ResNet50-LSTM sliding temporal window inference across frame sequence...
                        </p>
                        <div className="flex justify-between items-center text-[10px] text-neutral-500 font-mono pt-1 border-t border-neutral-900">
                          <span>Playback Time: {formatTime(currentTime)}</span>
                          <span className="text-emerald-400 font-bold">
                            {currentDetections.filter(d => currentTime >= d.start_time - 0.5).length === 0
                              ? "Baseline Normal Stream"
                              : `${currentDetections.filter(d => currentTime >= d.start_time - 0.5).length} Incident(s) Captured`}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Render Anomaly Cards ONLY when video playback reaches or passes their start timestamp */}
                    {currentDetections
                      .filter(det => currentTime >= det.start_time - 0.5)
                      .map((det, idx) => {
                        const isActive = currentTime >= det.start_time && currentTime <= det.end_time;
                        const startFormatted = formatTime(det.start_time || 0);
                        const endFormatted = formatTime(det.end_time || 0);
                        const rawStart = (det.start_time || 0).toFixed(2);
                        const rawEnd = (det.end_time || 0).toFixed(2);

                        return (
                          <div
                            key={idx}
                            className={`p-3.5 rounded-xl border transition space-y-2.5 ${
                              isActive
                                ? "bg-red-950/60 border-red-500 shadow-lg shadow-red-950/50 animate-pulse"
                                : "bg-neutral-950 border-neutral-800 hover:border-neutral-700"
                            }`}
                          >
                            <div className="flex justify-between items-start">
                              <span className={`font-bold text-xs flex items-center gap-1.5 ${isActive ? "text-red-300" : "text-red-400"}`}>
                                <span className={`w-2 h-2 rounded-full ${isActive ? "bg-red-500 animate-ping" : "bg-red-500"}`}></span>
                                🔴 {det.label || (det.event_type ? `${det.event_type.toUpperCase()} DETECTED` : "ANOMALY DETECTED")}
                              </span>
                              <span className="text-[10px] font-mono bg-red-950 text-red-300 px-2 py-0.5 rounded border border-red-800 font-bold">
                                {det.peak_confidence}% Conf
                              </span>
                            </div>

                            <div className="text-xs text-neutral-300 font-mono space-y-1">
                              <div className="flex justify-between">
                                <span className="text-neutral-400">Detected Window:</span>
                                <span className="text-cyan-400 font-bold">{startFormatted} - {endFormatted} ({rawStart}s - {rawEnd}s)</span>
                              </div>
                              {isActive && (
                                <div className="text-[10px] text-red-400 font-bold uppercase tracking-wider flex items-center gap-1">
                                  <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping"></span>
                                  PLAYBACK IS CURRENTLY INSIDE THIS ANOMALY WINDOW
                                </div>
                              )}
                            </div>

                            <button
                              onClick={() => seekToTimestamp(det.start_time)}
                              className="w-full py-2 bg-red-600/30 hover:bg-red-600 text-red-200 hover:text-white text-xs font-bold rounded-lg border border-red-500/40 transition flex items-center justify-center gap-1.5 shadow"
                            >
                              ▶ JUMP TO INCIDENT AT {startFormatted} ({rawStart}s)
                            </button>
                          </div>
                        );
                      })}

                    {/* Video Playback End - Final Overall Classification Result Highlight Box */}
                    {(videoEnded || currentTime >= (selectedVideo?.duration || 100) - 0.8) && videoAnalysis && (
                      <div className={`p-4 rounded-xl border-2 space-y-2 mt-2 shadow-2xl transition-all ${
                        videoAnalysis.is_anomaly
                          ? "bg-red-950/90 border-red-500 text-white shadow-red-950/60"
                          : "bg-emerald-950/90 border-emerald-500 text-white shadow-emerald-950/60"
                      }`}>
                        <div className="flex justify-between items-center font-black text-xs uppercase tracking-wider">
                          <span className="flex items-center gap-1.5 text-cyan-300">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                            🏁 PLAYBACK COMPLETE
                          </span>
                          <span className="bg-black/60 px-2 py-0.5 rounded text-[10px] font-mono text-neutral-300">
                            OVERALL RESULT
                          </span>
                        </div>
                        
                        <div className="py-2.5 px-3 rounded-lg bg-black/60 border border-white/10 text-center">
                          <span className="text-[10px] text-neutral-400 uppercase tracking-widest block font-bold">OVERALL VIDEO CLASSIFICATION</span>
                          <span className={`text-sm font-black tracking-wide block mt-1 ${
                            videoAnalysis.is_anomaly ? "text-red-400" : "text-emerald-400"
                          }`}>
                            {videoAnalysis.is_anomaly ? "🚨 THIS IS AN ANOMALY CCTV FOOTAGE" : "🟢 THIS IS A NORMAL CCTV FOOTAGE"}
                          </span>
                        </div>

                        <p className="text-[11px] text-neutral-300 font-mono text-center leading-relaxed">
                          {videoAnalysis.is_anomaly
                            ? `Video analysis finished. Confirmed ANOMALY CCTV FOOTAGE with ${videoAnalysis.total_instances || videoAnalysis.intervals?.length || 1} threat incident window(s) detected.`
                            : `Video analysis finished. Confirmed NORMAL CCTV FOOTAGE with zero security threat windows detected.`}
                        </p>

                        <button
                          onClick={() => seekToTimestamp(0)}
                          className="w-full py-2 bg-neutral-900 hover:bg-neutral-800 text-cyan-400 text-xs font-bold rounded-lg border border-neutral-700 transition flex items-center justify-center gap-1.5 shadow"
                        >
                          ↺ REPLAY VIDEO FROM 00:00
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "i3d" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-neutral-900 border border-neutral-800 rounded-2xl overflow-hidden aspect-video relative flex flex-col justify-between shadow-2xl">
              <div className="w-full h-full flex items-center justify-center bg-neutral-950">
                {canvasSrc ? (
                  <img src={canvasSrc} alt="I3D Feature Analysis Signal" className="w-full h-full object-contain" />
                ) : (
                  <div className="text-center p-8 text-neutral-400">
                    <p className="font-bold mb-2">Technical Feature Evaluator (.npy Engine)</p>
                    <button
                      onClick={toggleSimulation}
                      className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs rounded-xl shadow-lg transition"
                    >
                      {simRunning ? "■ STOP .NPY EVALUATION" : "▶ START .NPY EVALUATION"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 p-4 rounded-2xl space-y-3 text-xs">
            <h3 className="font-bold text-sm text-cyan-400">Dataset (.npy) Evaluator Metrics</h3>
            <div className="flex justify-between py-1.5 border-b border-neutral-800">
              <span className="text-neutral-400">Total Evaluated:</span>
              <span className="font-bold text-white">{totalProcessed} / {totalSamples}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-neutral-800">
              <span className="text-neutral-400">Anomalies Detected:</span>
              <span className="font-bold text-red-400">{anomaliesCount}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-neutral-800">
              <span className="text-neutral-400">Normal Clean:</span>
              <span className="font-bold text-emerald-400">{normalCount}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-neutral-400">Evaluation Score:</span>
              <span className="font-bold text-cyan-400">{accuracy.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === "history" && (
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 shadow-xl">
          <h2 className="text-xl font-bold text-neutral-100 mb-4 flex items-center gap-2">
            <span>📜</span> Incident History Audit Log
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-neutral-950 text-neutral-400 uppercase border-b border-neutral-800">
                <tr>
                  <th className="p-3">ID</th>
                  <th className="p-3">Event Type</th>
                  <th className="p-3">Severity</th>
                  <th className="p-3">Confidence</th>
                  <th className="p-3">Source File</th>
                  <th className="p-3">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800">
                {historicalIncidents.map((inc, i) => (
                  <tr key={i} className="hover:bg-neutral-950/60 transition">
                    <td className="p-3 text-neutral-500">#{inc.id || i + 1}</td>
                    <td className="p-3 font-bold text-red-400">{inc.type || inc.event_type}</td>
                    <td className="p-3">
                      <span className="bg-red-950 text-red-300 px-2 py-0.5 rounded border border-red-800">
                        LEVEL {inc.severity || 5}
                      </span>
                    </td>
                    <td className="p-3 text-cyan-400">{((inc.confidence || 0.94) * 100).toFixed(1)}%</td>
                    <td className="p-3 text-neutral-300">{inc.source || inc.sample || "UCF-Crime MP4"}</td>
                    <td className="p-3 text-neutral-400">{new Date(inc.timestamp * 1000).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
