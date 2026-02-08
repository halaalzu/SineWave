import { useState, useRef, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface GeminiChatProps {
  userId?: string;
}

export const GeminiChat = ({ userId = 'default_user' }: GeminiChatProps) => {
  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    content: "Hi! I'm your AI rehabilitation coach. I can answer questions about your hand movement data, progress, and exercises. What would you like to know?",
    timestamp: new Date()
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5001/api/gemini/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          user_id: userId
        })
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response || "I'm having trouble analyzing your data right now. Please try again.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      toast({
        title: "Error",
        description: "Failed to get AI response. Make sure Flask server is running.",
        variant: "destructive"
      });
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I'm having trouble connecting. Please make sure the server is running and try again.",
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="bg-card border-[6px] border-foreground shadow-card-retro flex flex-col h-[500px]">
      {/* Header */}
      <div className="bg-primary border-b-4 border-foreground p-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🤖</span>
          <div>
            <h3 className="font-display text-lg tracking-[0.2em] text-primary-foreground">
              AI COACH CHAT
            </h3>
            <p className="text-xs font-bold tracking-wider text-primary-foreground/70">
              POWERED BY GEMINI AI
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-3 border-3 border-foreground ${
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground'
              }`}
            >
              <div className="text-sm font-bold tracking-wide whitespace-pre-wrap">
                {msg.content}
              </div>
              <div className="text-[8px] font-bold tracking-wider opacity-50 mt-1">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-secondary text-secondary-foreground p-3 border-3 border-foreground">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t-4 border-foreground p-3 bg-muted">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your progress, exercises, or results..."
            disabled={isLoading}
            className="flex-1 bg-card border-3 border-foreground px-3 py-2 font-bold text-sm tracking-wide text-foreground placeholder:text-foreground/40 focus:outline-none focus:border-primary disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-primary border-3 border-foreground px-4 py-2 font-display text-sm tracking-widest text-primary-foreground shadow-btn hover:shadow-btn-hover active:shadow-btn-active transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            SEND
          </button>
        </div>
        <div className="mt-2 text-[8px] font-bold tracking-wider text-foreground/40">
          SUGGESTED: "How am I doing?" • "What should I focus on?" • "Explain my weak joints"
        </div>
      </div>
    </div>
  );
};
