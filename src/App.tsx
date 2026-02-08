import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Freestyle from "./pages/Freestyle";
import Levels from "./pages/Levels";
import LevelSession from "./pages/LevelSession";
import Analytics from "./pages/Analytics";
import HandComparison from "./pages/HandComparison";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/freestyle" element={<Freestyle />} />
          <Route path="/levels" element={<Levels />} />
          <Route path="/level/:levelId" element={<LevelSession />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/hand-comparison" element={<HandComparison />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
