import React from "react";
import "../Post.css";

interface PostProps {
  id: number;
  caption: string;
  videoUrl: string;
  likes: number;
  comments: number;
}

const Post: React.FC<PostProps> = ({
  id,
  caption,
  videoUrl,
  likes,
  comments,
}) => {
  return (
    <div className="post">
      <video className="video" controls>
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      <div className="post-details">
        <p>{caption}</p>
        <div className="stats">
          <span>❤️ {likes}</span>
          <span>💬 {comments}</span>
        </div>
      </div>
    </div>
  );
};

export default Post;
