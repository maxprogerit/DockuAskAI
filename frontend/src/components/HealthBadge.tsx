import { useEffect, useState } from "react";
import { api } from "../api";

type State = "checking" | "ok" | "degraded" | "down";

const LABEL: Record<State, string> = {
  checking: "Checking…",
  ok: "Backend online",
  degraded: "Backend degraded",
  down: "Backend unreachable",
};

export function HealthBadge() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const h = await api.health();
        if (!cancelled) setState(h.status === "ok" ? "ok" : "degraded");
      } catch {
        if (!cancelled) setState("down");
      }
    };
    void check();
    const timer = window.setInterval(check, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <span className={`badge badge-${state}`} role="status">
      <span className="dot" /> {LABEL[state]}
    </span>
  );
}
