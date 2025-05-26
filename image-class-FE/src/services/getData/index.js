import { useMutation } from '@tanstack/react-query';

const fetchRootData = async () => {
  const response = await fetch(`${import.meta.env.VITE_API_URL}/`);
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  return response.json();
};

const fetchSearchResults = async (searchQuery) => {
  const response = await fetch(`${import.meta.env.VITE_API_URL}/search?q=${encodeURIComponent(searchQuery)}`);
  if (!response.ok) {
    const errorData = await response.text(); 
    throw new Error(`Network response was not ok: ${response.status} ${response.statusText}. ${errorData}`);
  }
  return response.json();
};

export const useGetData = () => {
  return useMutation({ mutationFn: fetchRootData });
};

export const useSearchMutation = () => {
  return useMutation({
    mutationFn: fetchSearchResults,
  });
};