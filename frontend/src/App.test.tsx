import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from './App';
import * as api from './api';

// Mock the API module
vi.mock('./api');

describe('App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    const { container } = render(<App />);
    expect(container).toBeTruthy();
  });

  it('displays main heading', () => {
    render(<App />);
    expect(screen.getByText('Visual Search Engine')).toBeInTheDocument();
  });

  it('has index image section', () => {
    render(<App />);
    expect(screen.getByText('Index Image')).toBeInTheDocument();
  });

  it('has search section', () => {
    render(<App />);
    expect(screen.getByText('Search')).toBeInTheDocument();
  });

  describe('File Input', () => {
    it('accepts file input', () => {
      render(<App />);
      const fileInput = screen.getAllByType('file')[0];
      expect(fileInput).toBeInTheDocument();
    });

    it('updates file state when file is selected', () => {
      render(<App />);
      const fileInput = screen.getAllByType('file')[0];

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      fireEvent.change(fileInput, { target: { files: [file] } });

      // Button should be enabled after file selection
      const indexButton = screen.getByText('Index Image');
      expect(indexButton).not.toBeDisabled();
    });
  });

  describe('Text Search', () => {
    it('has text input and search button', () => {
      render(<App />);
      const textInput = screen.getByPlaceholderText('Search by text');
      const searchButton = screen.getByText('Search by Text');

      expect(textInput).toBeInTheDocument();
      expect(searchButton).toBeInTheDocument();
    });

    it('enables search button when query is entered', () => {
      render(<App />);
      const textInput = screen.getByPlaceholderText('Search by text');
      const searchButton = screen.getByText('Search by Text');

      fireEvent.change(textInput, { target: { value: 'sunset' } });

      expect(searchButton).not.toBeDisabled();
    });

    it('disables search button when query is empty', () => {
      render(<App />);
      const searchButton = screen.getByText('Search by Text');

      expect(searchButton).toBeDisabled();
    });
  });

  describe('Loading State', () => {
    it('shows loading indicator during API calls', async () => {
      vi.mocked(api.searchByText).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve([]), 100))
      );

      render(<App />);
      const textInput = screen.getByPlaceholderText('Search by text');
      const searchButton = screen.getByText('Search by Text');

      fireEvent.change(textInput, { target: { value: 'test' } });
      fireEvent.click(searchButton);

      expect(screen.getByText('Loading...')).toBeInTheDocument();

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });
    });
  });

  describe('Results Display', () => {
    it('displays search results', async () => {
      const mockResults = [
        {
          imageId: 'test1.jpg',
          description: 'A beautiful sunset',
          image_url: 'http://example.com/test1.jpg'
        },
        {
          imageId: 'test2.jpg',
          description: 'A mountain landscape',
          image_url: 'http://example.com/test2.jpg'
        }
      ];

      vi.mocked(api.searchByText).mockResolvedValue(mockResults);

      render(<App />);
      const textInput = screen.getByPlaceholderText('Search by text');
      const searchButton = screen.getByText('Search by Text');

      fireEvent.change(textInput, { target: { value: 'sunset' } });
      fireEvent.click(searchButton);

      await waitFor(() => {
        expect(screen.getByText('A beautiful sunset')).toBeInTheDocument();
        expect(screen.getByText('A mountain landscape')).toBeInTheDocument();
      });
    });

    it('displays empty state when no results', async () => {
      vi.mocked(api.searchByText).mockResolvedValue([]);

      render(<App />);
      const textInput = screen.getByPlaceholderText('Search by text');
      const searchButton = screen.getByText('Search by Text');

      fireEvent.change(textInput, { target: { value: 'nonexistent' } });
      fireEvent.click(searchButton);

      await waitFor(() => {
        expect(screen.getByText('No results found.')).toBeInTheDocument();
      });
    });
  });

  describe('Image Indexing', () => {
    it('displays caption after successful indexing', async () => {
      const mockResponse = { caption: 'A serene beach scene' };
      vi.mocked(api.uploadImage).mockResolvedValue(mockResponse);

      render(<App />);
      const fileInput = screen.getAllByType('file')[0];
      const indexButton = screen.getByText('Index Image');

      const file = new File(['test'], 'beach.jpg', { type: 'image/jpeg' });
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(indexButton);

      await waitFor(() => {
        expect(screen.getByText('Caption: A serene beach scene')).toBeInTheDocument();
      });
    });

    it('shows success message after indexing', async () => {
      vi.mocked(api.uploadImage).mockResolvedValue({ caption: 'Test caption' });

      render(<App />);
      const fileInput = screen.getAllByType('file')[0];
      const indexButton = screen.getByText('Index Image');

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(indexButton);

      await waitFor(() => {
        expect(screen.getByText('Image indexed successfully!')).toBeInTheDocument();
      });
    });

    it('shows error message on indexing failure', async () => {
      vi.mocked(api.uploadImage).mockRejectedValue(new Error('Upload failed'));

      render(<App />);
      const fileInput = screen.getAllByType('file')[0];
      const indexButton = screen.getByText('Index Image');

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(indexButton);

      await waitFor(() => {
        expect(screen.getByText('Failed to index image.')).toBeInTheDocument();
      });
    });
  });

  describe('Image Search', () => {
    it('performs image search', async () => {
      const mockResults = [
        {
          imageId: 'similar.jpg',
          description: 'Similar image',
          image_url: 'http://example.com/similar.jpg'
        }
      ];

      vi.mocked(api.searchByImage).mockResolvedValue(mockResults);

      render(<App />);
      const fileInputs = screen.getAllByType('file');
      const searchByImageButton = screen.getByText('Search by Image');

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      fireEvent.change(fileInputs[1], { target: { files: [file] } });
      fireEvent.click(searchByImageButton);

      await waitFor(() => {
        expect(screen.getByText('Similar image')).toBeInTheDocument();
      });
    });
  });
});

