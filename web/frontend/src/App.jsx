import { useRef, useEffect } from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";

const ImageUploader = (props) => {
  const inputEl = useRef();
  const handleDragOver = (e) => {
    console.log("drag over");
  };

  useEffect(() => {
    const drop_region = document.getElementById("drop-region");
    [("dragenter", "dragover")].forEach((eventName) => {
      drop_region.addEventListener(eventName, (e) => {
        // e.stopPropagation();
        // console.log("drag over");
        drop_region.classList.add("drag-over");
        return false;
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      drop_region.addEventListener(eventName, (e) => {
        e.preventDefault();
        drop_region.classList.remove("drag-over");

        return false;
      });
    });
    drop_region.addEventListener("click", () => {
      inputEl.current.click();
    });
    drop_region.onDrop = (e) => {
      e.topPropagation();
      console.log("drop");
      e.preventDefault();
    };
  }, []);
  const handleDrop = (e) => {
    e.topPropagation();
    console.log("drop");
    e.preventDefault();
    return false;
  };
  return (
    <div
      id="drop-region"
      className="image-view uploader text-center"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <h2>Drag and Drop or Click to Upload</h2>
      <input
        id="file-input"
        type="file"
        multiple
        ref={inputEl}
        // onChange={this.handleInputByClick}
      />
    </div>
  );
};

const ImageDisplayer = (props) => {
  return (
    <div className="image-view text-center">
      <h2>Output</h2>
    </div>
  );
};

const App = () => {
  const uploadFiles = (files) => {
    console.log(files);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    console.log("drop");
    const files = e.dataTransfer.files;
    uploadFiles(Array.from(files));
    return false;
  };

  return (
    <div className="App">
      <div className="container">
        <h1>hi</h1>
        <div className="row">
          <div
            className="col uploader-col"
            onDrop={() => {
              console.log("drop");
            }}
          >
            <ImageUploader
              type="Input Low Resolution Image"
              handleDrop={handleDrop}
            />
          </div>
          <div className="col text-center">
            <button className="btn btn-primary">Convert</button>
          </div>
          <div className="col uploader-col">
            <ImageDisplayer />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
