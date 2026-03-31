import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { searchByText, uploadImage, searchByImage } from './api';

// Mock axios
vi.mock('axios');

describe('API Functions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('searchByText', () => {
    it('should call axios.get with correct parameters', async () => {
      const mockResponse = {
        data: [
          {
            imageId: 'test.jpg',
            description: 'A test image',
            imageUrl: 'http://example.com/test.jpg'
          }
        ]
      };

      vi.mocked(axios.get).mockResolvedValue(mockResponse);

      const result = await searchByText('sunset');

      expect(axios.get).toHaveBeenCalledWith(
        'http://localhost:8000/search-by-text/',
        { params: { query: 'sunset' } }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should handle API errors', async () => {
      const mockError = new Error('Network error');
      vi.mocked(axios.get).mockRejectedValue(mockError);

      await expect(searchByText('test')).rejects.toThrow('Network error');
    });

    it('should return empty array when no results found', async () => {
      vi.mocked(axios.get).mockResolvedValue({ data: [] });

      const result = await searchByText('nonexistent');

      expect(result).toEqual([]);
    });
  });

  describe('uploadImage', () => {
    it('should upload image with FormData', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockResponse = { data: { caption: 'A beautiful sunset' } };

      vi.mocked(axios.post).mockResolvedValue(mockResponse);

      const result = await uploadImage(mockFile);

      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:8000/index-image/',
        expect.any(FormData)
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should handle upload errors', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockError = new Error('Upload failed');
      vi.mocked(axios.post).mockRejectedValue(mockError);

      await expect(uploadImage(mockFile)).rejects.toThrow('Upload failed');
    });

    it('should append file to FormData correctly', async () => {
      const mockFile = new File(['test content'], 'image.jpg', { type: 'image/jpeg' });
      const mockResponse = { data: { caption: 'Test caption' } };

      vi.mocked(axios.post).mockImplementation((url, data) => {
        // Verify FormData contains the file
        expect(data instanceof FormData).toBe(true);
        return Promise.resolve(mockResponse);
      });

      await uploadImage(mockFile);
    });
  });

  describe('searchByImage', () => {
    it('should search by image with FormData', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockResponse = {
        data: [
          {
            imageId: 'similar.jpg',
            description: 'A similar image',
            imageUrl: 'http://example.com/similar.jpg'
          }
        ]
      };

      vi.mocked(axios.post).mockResolvedValue(mockResponse);

      const result = await searchByImage(mockFile);

      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:8000/search-by-image/',
        expect.any(FormData)
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should handle search errors', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockError = new Error('Search failed');
      vi.mocked(axios.post).mockRejectedValue(mockError);

      await expect(searchByImage(mockFile)).rejects.toThrow('Search failed');
    });

    it('should return empty array when no similar images found', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      vi.mocked(axios.post).mockResolvedValue({ data: [] });

      const result = await searchByImage(mockFile);

      expect(result).toEqual([]);
    });
  });

  describe('Error Handling', () => {
    it('should handle timeout errors', async () => {
      const timeoutError = new Error('timeout of 5000ms exceeded');
      vi.mocked(axios.get).mockRejectedValue(timeoutError);

      await expect(searchByText('test')).rejects.toThrow();
    });

    it('should handle 500 server errors', async () => {
      const serverError = {
        response: { status: 500, data: { message: 'Internal server error' } }
      };
      vi.mocked(axios.get).mockRejectedValue(serverError);

      await expect(searchByText('test')).rejects.toThrow();
    });
  });
});
