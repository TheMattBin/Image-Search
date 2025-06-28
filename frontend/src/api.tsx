import axios from "axios";

export async function searchByText(query: string) {
  const res = await axios.get("http://localhost:8000/search-by-text/", {
    params: { query }
  });
  return res.data;
}

export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append("image", file);
  const res = await axios.post("http://localhost:8000/index-image/", formData);
  return res.data;
}

export async function searchByImage(file: File) {
  const formData = new FormData();
  formData.append("image", file);
  const res = await axios.post("http://localhost:8000/search-by-image/", formData);
  return res.data;
}