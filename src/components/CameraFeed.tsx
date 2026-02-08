import React, { useState } from 'react';

interface CameraFeedProps {
  className?: string;
}

export const CameraFeed: React.FC<CameraFeedProps> = ({ className = '' }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const handleImageLoad = () => {
    setIsLoading(false);
    setHasError(false);
  };

  const handleImageError = () => {
    setIsLoading(false);
    setHasError(true);
  };

  return (
    <div className={`relative ${className}`}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="text-white">Loading camera...</div>
        </div>
      )}
      {hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="text-white text-center">
            <p>Camera feed unavailable</p>
            <p className="text-sm mt-2">Make sure Flask server is running on port 5001</p>
          </div>
        </div>
      )}
      <img
        src="http://localhost:5001/video_feed"
        alt="Hand Tracking Camera"
        onLoad={handleImageLoad}
        onError={handleImageError}
        className="w-full h-full object-cover rounded-lg"
      />
    </div>
  );
};
