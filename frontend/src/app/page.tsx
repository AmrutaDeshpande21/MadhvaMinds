"use client";

import { useEffect, useState, useRef } from "react";
import Head from "next/head";

export default function Home() {
  const [activeTab, setActiveTab] = useState("live"); // 'live' or 'history'
  const [alerts, setAlerts] = useState<any[]>([]);
  const [historicalIncidents, setHistoricalIncidents] = useState<any[]>([]);
  const [videoSrc, setVideoSrc] = useState<string>("");
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
  }, [activeTab]);

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
