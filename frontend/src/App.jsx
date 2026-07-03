import { Routes, Route } from "react-router-dom";
import "./App.css";
import { WorkspaceProvider } from "./context/WorkspaceProvider";
import HomePage from "./pages/HomePage";
import DescriptionPage from "./pages/DescriptionPage";
import LabelPage from "./pages/LabelPage";

export default function App() {
  return (
    <WorkspaceProvider>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/description" element={<DescriptionPage />} />
        <Route path="/label" element={<LabelPage />} />
      </Routes>
    </WorkspaceProvider>
  );
}
