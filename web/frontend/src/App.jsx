import { useRef, useState, useEffect, Fragment } from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";

const ImageUploader = (props) => {
  const inputEl = useRef();
  const [imgsrc, setImgSrc] = useState(null);
  useEffect(() => {
    const drop_region = document.getElementById("drop-region");
    function handle_drop(e) {
      e.preventDefault();
      e.stopPropagation();
      drop_region.classList.remove("drag-over");
      props.handleDrop(e, setImgSrc);
    }

    ["dragenter", "dragover"].forEach((eventName) => {
      drop_region.addEventListener(
        eventName,
        (e) => {
          e.preventDefault();
          e.stopPropagation();
          drop_region.classList.add("drag-over");
        },
        false
      );
    });
    ["dragleave", "drop"].forEach((eventName) => {
      drop_region.addEventListener(eventName, handle_drop, false);
    });
    drop_region.addEventListener("click", () => {
      inputEl.current.click();
    });
    drop_region.onDrop = (e) => {
      e.topPropagation();
      e.preventDefault();
    };
  }, []);
  return (
    <div id="drop-region" className="image-view uploader text-center">
      {imgsrc ? (
        <img src={imgsrc} alt="" />
      ) : (
        <Fragment>
          <h2>Drag and Drop or Click to Upload</h2>
        </Fragment>
      )}
      <input
        id="file-input"
        type="file"
        ref={inputEl}
        onChange={(e) => {
          props.handleInputByClick(e, setImgSrc);
        }}
      />
    </div>
  );
};

const ImageDisplayer = (props) => {
  return (
    <div className="image-view text-center">
      {props.img ? <img src={props.img} /> : <h2>Output</h2>}
    </div>
  );
};

const App = () => {
  const [imgFile, setImgFile] = useState(null);
  const [weight, setWeight] = useState(null);
  const [imgdisplay, setImgdisplay] = useState(null);
  const handleFileInput = (file, setImgSrc) => {
    const formData = new FormData();
    if (FileReader && file) {
      var fr = new FileReader();
      fr.onload = function () {
        setImgSrc(fr.result);
      };
      fr.readAsDataURL(file);
    }
    setImgFile(file);
  };

  const handleDrop = (e, setImgSrc) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFileInput(Array.from(files)[0], setImgSrc);
    return false;
  };

  const handleInputByClick = (e, setImgSrc) => {
    handleFileInput(e.target.files[0], setImgSrc);
  };

  const handleConvert = (e) => {
    const data = new FormData();
    data.append("image", imgFile);
    data.append("weight", weight);
    axios
      .post("/sr", data)
      .then((res) => {
        setImgdisplay("data:image/png;base64," + res.data);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const inputWeight = (e) => {
    setWeight(e.target.files[0]);
  };

  return (
    <div className="App">
      <div className="container">
        <h1>Super Resolution</h1>
        <div className="row">
          <div id="drop-region-container" className="col uploader-col">
            <ImageUploader
              type="Input Low Resolution Image"
              handleDrop={handleDrop}
              handleInputByClick={handleInputByClick}
            />
          </div>
          <div className="col text-center">
            <button className="btn btn-primary" onClick={handleConvert}>
              Convert
            </button>
          </div>
          <div className="col uploader-col">
            <ImageDisplayer img={imgdisplay} />
          </div>
        </div>
        <br />
        <input
          className="form-control"
          type="file"
          name="weight-input"
          id="model-weight-input"
          onChange={inputWeight}
        />
      </div>
    </div>
  );
};

export default App;
