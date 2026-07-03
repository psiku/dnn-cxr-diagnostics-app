import { useCallback, useMemo, useReducer } from "react";
import { WorkspaceContext } from "./workspaceContext";

const initialState = {
  studySessionId: "",
  studyFile: null,
  studyFileName: "",
  originalDataUrl: "",
  heatmapBase64: "",
};

function workspaceReducer(state, action) {
  switch (action.type) {
    case "resetStudyWithUpload":
      return {
        ...state,
        studySessionId: crypto.randomUUID(),
        studyFile: action.file,
        studyFileName: action.file?.name ?? "study.jpg",
        originalDataUrl: action.originalDataUrl,
        heatmapBase64: "",
      };
    case "setHeatmapBase64":
      return { ...state, heatmapBase64: action.heatmapBase64 ?? "" };
    default:
      return state;
  }
}

export function WorkspaceProvider({ children }) {
  const [exam, dispatch] = useReducer(workspaceReducer, initialState);

  const patchExamFromUpload = useCallback((file, originalDataUrl) => {
    if (!file) return;
    dispatch({ type: "resetStudyWithUpload", file, originalDataUrl });
  }, []);

  const setExamHeatmap = useCallback((heatmapBase64) => {
    dispatch({ type: "setHeatmapBase64", heatmapBase64 });
  }, []);

  const value = useMemo(
    () => ({
      exam,
      patchExamFromUpload,
      setExamHeatmap,
      hasExam: Boolean(exam.studyFile && exam.originalDataUrl),
    }),
    [exam, patchExamFromUpload, setExamHeatmap]
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
