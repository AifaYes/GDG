import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import Header from "./components/header";

interface TikTokPost {
  caption: string;
  image_url: string;
  video_url: string;
  likes: number;
  comments: number;
}

const API_KEY = "59c69daf081542c687877fa24b741edd"; // Replace with your actual API key

const App: React.FC = () => {
  const [post, setPost] = useState<TikTokPost | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isButtonClicked, setIsButtonClicked] = useState(false);

  const generatePost = async () => {
    try {
      setError(null);
      setPost(null);

      const response = await axios.post(
        "https://57cb-34-125-125-47.ngrok-free.app", // Replace with your ngrok URL
        {},
        {
          headers: {
            Authorization: `Bearer ${API_KEY}`,
            "Content-Type": "application/json",
          },
        }
      );

      setPost(response.data);
      setIsButtonClicked(true);
    } catch (error) {
      console.error("Error generating post:", error);
      setError("An error occurred while generating the post.");
    }
  };

  return (
    <div className="app">
      <div
        className={`header-block ${isButtonClicked ? "move-to-corner" : ""}`}
      >
        <Header />
        <button onClick={generatePost}>Generate Trendy Posts</button>
      </div>

      {error && <p className="error">{error}</p>}

      {post && (
        <div className="post">
          {post.video_url ? (
            <video className="video" controls>
              <source src={post.video_url} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          ) : (
            <img src={post.image_url} alt="Post" className="image" />
          )}
          <div className="post-details">
            <p>{post.caption}</p>
            <div className="stats">
              <span>❤️ {post.likes}</span>
              <span>💬 {post.comments}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
