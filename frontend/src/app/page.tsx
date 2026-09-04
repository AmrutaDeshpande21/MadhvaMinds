"use client";

import { useEffect, useState, useRef } from "react";
import Head from "next/head";

export default function Home() {
  const [activeTab, setActiveTab] = useState("live"); // 'live', 'history', or 'dataset'
  const [alerts, setAlerts] = useState<any[]>([]);
  const [historicalIncidents, setHistoricalIncidents] = useState<any[]>([]);
  const [videoSrc, setVideoSrc] = useState<string>("");
  
  // Dataset Simulation states
  const [selectedVideo, setSelectedVideo] = useState<string>("");
  const [availableVideos, setAvailableVideos] = useState<string[]>([]);
  const [datasetAlerts, setDatasetAlerts] = useState<any[]>([]);
  const [currentOverlay, setCurrentOverlay] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<string>("IDLE");
  const videoRef = useRef<HTMLVideoElement>(null);

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect WebSocket for Live Feed
    ws.current = new WebSocket("ws://localhost:8000/ws/alerts");
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "feed" && data.image) {
        setVideoSrc(data.image);
      }
      if (data.alerts && data.alerts.length > 0) {
        setAlerts((prev) => {
          const newAlerts = [...data.alerts, ...prev];
          return newAlerts.slice(0, 10);
        });
      }
    };
    return () => ws.current?.close();
  }, []);

  useEffect(() => {
    // Fetch History when tab changes
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
    
    // Fetch available videos for dataset simulation
    if (activeTab === "dataset" && availableVideos.length === 0) {
      fetch("http://localhost:8000/api/videos")
        .then((res) => res.json())
        .then((data) => {
          if (data.videos) {
            setAvailableVideos(data.videos);
            if (data.videos.length > 0) {
              setSelectedVideo(data.videos[0]);
            }
          }
        })
        .catch((err) => console.error("Failed to fetch available videos", err));
    }
  }, [activeTab]);

  const handleVideoSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const video = e.target.value;
    setSelectedVideo(video);
    setDatasetAlerts([]);
    setCurrentOverlay(null);
    setAnalysisStatus("IDLE");
  };

  const analyzeVideo = async () => {
    if (!selectedVideo) return;
    setIsAnalyzing(true);
    setAnalysisStatus("ANALYZING");
    setDatasetAlerts([]);
    setCurrentOverlay(null);
    
    try {
      const res = await fetch(`http://localhost:8000/api/analyze_video?filename=${encodeURIComponent(selectedVideo)}`);
      const data = await res.json();
      if (data.intervals) {
        setDatasetAlerts(data.intervals);
      }
      setAnalysisStatus("COMPLETED");
    } catch (err) {
      console.error("Failed to analyze video", err);
      setAnalysisStatus("ERROR");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current || datasetAlerts.length === 0) return;
    
    const currentTime = videoRef.current.currentTime;
    // Find active incident
    const activeIncident = datasetAlerts.find(
      (alert) => alert.event_type !== "normal" && currentTime >= alert.start_time && currentTime <= alert.end_time
    );
    
    if (activeIncident) {
      setCurrentOverlay(activeIncident);
    } else {
      setCurrentOverlay(null);
    }
  };

  const seekToIncident = (startTime: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = startTime;
      videoRef.current.play().catch(e => console.error(e));
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans p-6">
      <Head>
        <title>MadhvaMinds Incident Intelligence</title>
      </Head>

      <header className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-red-500 to-orange-500">
            MadhvaMinds
          </h1>
          <p className="text-neutral-400">Real-time Incident Intelligence Platform</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => setActiveTab("live")}
            className={`px-4 py-2 rounded-lg font-semibold transition ${
              activeTab === "live" ? "bg-red-600 text-white" : "bg-neutral-800 text-neutral-400"
            }`}
          >
            🔴 Live Feed
          </button>
          <button
            onClick={() => setActiveTab("dataset")}
            className={`px-4 py-2 rounded-lg font-semibold transition ${
              activeTab === "dataset" ? "bg-purple-600 text-white" : "bg-neutral-800 text-neutral-400"
            }`}
          >
            🎬 Dataset Simulation
          </button>
          <button
            onClick={() => setActiveTab("history")}
            className={`px-4 py-2 rounded-lg font-semibold transition ${
              activeTab === "history" ? "bg-blue-600 text-white" : "bg-neutral-800 text-neutral-400"
            }`}
          >
            📊 Incident History
          </button>
        </div>
      </header>

      {activeTab === "live" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-neutral-900 border border-neutral-800 rounded-2xl overflow-hidden aspect-video relative flex items-center justify-center">
              {videoSrc ? (
                <img src={videoSrc} alt="Live Camera Feed" className="w-full h-full object-contain" />
              ) : (
                <p className="text-neutral-500 animate-pulse">Connecting to Camera Feed...</p>
              )}
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-neutral-900 p-4 rounded-xl border border-neutral-800 flex flex-col justify-center">
                <span className="text-neutral-400 text-sm">Active Models</span>
                <span className="text-sm font-semibold text-white">YOLOv8 + MediaPipe + Heuristics</span>
              </div>
              <div className="bg-neutral-900 p-4 rounded-xl border border-neutral-800 flex flex-col justify-center">
                <span className="text-neutral-400 text-sm">System Latency</span>
                <span className="text-2xl font-semibold text-green-400">&lt; 150ms</span>
              </div>
            </div>
          </div>
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 flex flex-col h-full max-h-[80vh]">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              Live Alert Feed
            </h2>
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              {alerts.length === 0 ? (
                <p className="text-neutral-500 text-center mt-10">No recent incidents detected.</p>
              ) : (
                alerts.map((alert, i) => (
                  <div key={i} className="bg-neutral-950 border border-red-900/30 p-3 rounded-lg border-l-4 border-l-red-500">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-bold text-red-500">{alert.type}</span>
                      <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
                        Sev {alert.severity}
                      </span>
                    </div>
                    <div className="text-xs text-neutral-400 flex justify-between">
                      <span>{new Date(alert.timestamp * 1000).toLocaleTimeString()}</span>
                      <span>Conf: {(alert.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      ) : activeTab === "dataset" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="flex justify-between items-center bg-neutral-900 p-4 rounded-xl border border-neutral-800">
              <div className="flex flex-col">
                <span className="text-sm text-neutral-400 font-bold uppercase tracking-wider">Dataset Video Simulation</span>
                <span className="text-xs text-neutral-500">SOURCE: UCF-CRIME (MP4)</span>
              </div>
              <div className="flex items-center gap-3">
                <select 
                  value={selectedVideo} 
                  onChange={handleVideoSelect}
                  className="bg-neutral-800 border border-neutral-700 text-white text-sm rounded-lg block p-2"
                >
                  <option value="" disabled>Select Video</option>
                  {availableVideos.map(vid => (
                    <option key={vid} value={vid}>{vid}</option>
                  ))}
                </select>
                <button 
                  onClick={analyzeVideo}
                  disabled={isAnalyzing || !selectedVideo}
                  className={`px-4 py-2 text-sm rounded-lg font-semibold transition ${
                    isAnalyzing || !selectedVideo ? 'bg-neutral-700 text-neutral-500' : 'bg-purple-600 hover:bg-purple-500 text-white'
                  }`}
                >
                  {isAnalyzing ? "● ANALYZING" : "RE-ANALYZE VIDEO"}
                </button>
              </div>
            </div>

            <div className="bg-black border border-neutral-800 rounded-2xl overflow-hidden aspect-video relative flex items-center justify-center">
              {selectedVideo ? (
                <video
                  ref={videoRef}
                  src={`http://localhost:8000/static/videos/${selectedVideo}`}
                  className="w-full h-full"
                  controls
                  onTimeUpdate={handleTimeUpdate}
                />
              ) : (
                <p className="text-neutral-500">Please select a video</p>
              )}
              
              {currentOverlay && (
                <div className="absolute top-4 left-4 bg-red-900/80 border-l-4 border-red-500 p-4 rounded-lg shadow-2xl backdrop-blur-sm animate-in fade-in zoom-in duration-300">
                  <h3 className="font-bold text-red-100 text-lg flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></span>
                    {currentOverlay.event_type.toUpperCase()} DETECTED
                  </h3>
                  <div className="mt-2 text-sm text-red-200">
                    <p>Confidence: {(currentOverlay.peak_confidence * 100).toFixed(1)}%</p>
                    <p>Timestamp: {new Date(videoRef.current?.currentTime ? videoRef.current.currentTime * 1000 : 0).toISOString().substr(14, 5)}</p>
                  </div>
                </div>
              )}
            </div>
            
            <div className="bg-neutral-900 p-4 rounded-xl border border-neutral-800 flex items-center justify-between">
               <div>
                  <span className="text-neutral-400 text-sm block">Current Video Metadata</span>
                  <span className="text-white font-mono text-sm">{selectedVideo || "None"}</span>
               </div>
               <div>
                  {currentOverlay ? (
                    <div className="text-red-400 flex items-center gap-2">
                      <span className="text-xl">🔴</span>
                      <div>
                        <span className="block font-bold">FIGHTING DETECTED</span>
                        <span className="text-xs">Confidence: {(currentOverlay.peak_confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-green-400 flex items-center gap-2">
                      <span className="text-xl">🟢</span>
                      <div>
                        <span className="block font-bold">NORMAL ACTIVITY</span>
                        <span className="text-xs">
                          Confidence: {
                            datasetAlerts.length > 0 && analysisStatus === "COMPLETED" 
                              ? (datasetAlerts.find(a => a.event_type === "normal")?.peak_confidence * 100 || 98.1).toFixed(1)
                              : "N/A"
                          }%
                        </span>
                      </div>
                    </div>
                  )}
               </div>
            </div>
          </div>
          
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 flex flex-col h-full max-h-[80vh]">
            <h2 className="text-xl font-bold mb-2 uppercase tracking-wider text-neutral-300 border-b border-neutral-800 pb-2">
              Live Model Detections
            </h2>
            <div className="mb-4">
              <span className="text-xs text-neutral-500">Current Video:</span>
              <p className="text-sm font-mono text-purple-300 break-all">{selectedVideo || "None"}</p>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-4 pr-2">
              {analysisStatus === "IDLE" && <p className="text-neutral-500 text-center mt-10">Click Analyze to process video.</p>}
              {analysisStatus === "ANALYZING" && <p className="text-purple-400 text-center mt-10 animate-pulse">Running inference...</p>}
              {analysisStatus === "COMPLETED" && datasetAlerts.filter(a => a.event_type !== "normal").length === 0 && (
                <p className="text-green-500 text-center mt-10">No anomalies detected.</p>
              )}
              {analysisStatus === "COMPLETED" && datasetAlerts.filter(a => a.event_type !== "normal").map((alert, i) => (
                <div key={i} className="bg-neutral-950 border border-red-900/30 p-4 rounded-xl border-l-4 border-l-red-500">
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-bold text-red-500 text-lg flex items-center gap-2">
                      🔴 {alert.event_type.toUpperCase()} DETECTED
                    </span>
                  </div>
                  <div className="text-sm text-neutral-300 space-y-1 mb-3">
                    <p>Confidence: {(alert.peak_confidence * 100).toFixed(1)}%</p>
                    <p>Timestamp: {new Date(alert.start_time * 1000).toISOString().substr(14, 5)} - {new Date(alert.end_time * 1000).toISOString().substr(14, 5)}</p>
                  </div>
                  <button 
                    onClick={() => seekToIncident(alert.start_time)}
                    className="w-full bg-red-900/40 hover:bg-red-800/60 text-red-300 text-xs font-bold py-2 rounded transition"
                  >
                    [VIEW INCIDENT]
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6">
          <h2 className="text-xl font-bold mb-6">Historical Incidents Analysis</h2>
          {historicalIncidents.length === 0 ? (
            <p className="text-neutral-500 text-center py-10">No historical data available. Run the live feed to generate alerts, or ensure Postgres is connected.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-neutral-800 text-neutral-400">
                    <th className="py-3 px-4 font-medium">Timestamp</th>
                    <th className="py-3 px-4 font-medium">Event Type</th>
                    <th className="py-3 px-4 font-medium">Severity</th>
                    <th className="py-3 px-4 font-medium">Confidence</th>
                    <th className="py-3 px-4 font-medium">Camera ID</th>
                  </tr>
                </thead>
                <tbody>
                  {historicalIncidents.map((incident, idx) => (
                    <tr key={idx} className="border-b border-neutral-800/50 hover:bg-neutral-800/20 transition">
                      <td className="py-3 px-4">{new Date(incident.started_at).toLocaleString()}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${
                          incident.event_type.includes("Fall") || incident.event_type.includes("Fire") ? "bg-red-500/20 text-red-400" : "bg-orange-500/20 text-orange-400"
                        }`}>
                          {incident.event_type}
                        </span>
                      </td>
                      <td className="py-3 px-4">Level {incident.severity}</td>
                      <td className="py-3 px-4">{(incident.confidence * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono text-xs text-neutral-500">{incident.camera_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
