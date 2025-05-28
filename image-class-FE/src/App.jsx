import { useState, useEffect } from 'react';
import { IconArrowRight, IconSearch, IconUpload, IconInfoCircle, IconDownload } from '@tabler/icons-react'; 
import { ActionIcon, TextInput, useMantineTheme, Container, Loader, SimpleGrid, Group, Modal, Image, Text, AspectRatio, Grid, Stack, Title } from '@mantine/core'; 
import { notifications } from '@mantine/notifications';
import { useDebouncedCallback } from '@mantine/hooks';
import './App.css';
import { useSearchMutation } from './services/getData';

const fileType = (url_raw) => {
  const url = url_raw.toLowerCase();
  if (url.endsWith('.jpg') || url.endsWith('.jpeg')) return 'JPEG';
  if (url.endsWith('.png')) return 'PNG';
  if (url.endsWith('.gif')) return 'GIF';
  if (url.endsWith('.webp')) return 'WebP';

  return 'Unknown';
}
function App() {
  const theme = useMantineTheme();
  const [inputValue, setInputValue] = useState('');
  const [selectedItem, setSelectedItem] = useState(null);
  const [modalOpened, setModalOpened] = useState(false);

  const { mutate: searchMutate, data: searchResult, isPending: isSearchPending, error: searchError } = useSearchMutation();

  console.log(searchResult)
  useEffect(() => {
    if (!isSearchPending && !searchError && searchResult?.length > 0) {
      notifications.show({
        title: 'Search Complete',
        message: `${searchResult.length} image${searchResult.length !== 1 ? 's' : ''} matching your query found.`,
        color: 'teal',
        icon: <IconInfoCircle size={18} />,
        autoClose: 5000, 
      });
    }
  }, [searchResult, isSearchPending, searchError]); 

  const executeSearch = (value) => {
    console.log('Executing debounced search for:', value);
    if (value.trim() !== '') {
      searchMutate(value);
    }
  };

  const debouncedExecuteSearch = useDebouncedCallback(executeSearch, 1000);

  const handleImageSearch = (event) => {
    const currentValue = event.currentTarget.value;
    setInputValue(currentValue);
    debouncedExecuteSearch(currentValue);
  };

  const handleUploadClick = () => {
    console.log('Upload button clicked');
    // TODO: Implement upload functionality
  };

  const handleItemClick = (item) => {
    console.log('handleItemClick called with item:', item);
    setSelectedItem(item);
    setModalOpened(true);
    console.log('After setting state: selectedItem =', item, ', modalOpened = true');
  };

  const handleDownloadImage = async (event, item) => {
    event.stopPropagation(); // Prevent modal from opening when download icon is clicked
    if (!item || !item.image_url) {
      console.error("No image URL available for download", item);
      // Optionally, add a user notification here (e.g., using Mantine's notification system)
      return;
    }

    const imageUrl = `${import.meta.env.VITE_API_URL}${encodeURI(selectedItem.image_url)}`;
    // Attempt to get a reasonable filename
    let fileName = item.name || item.unique_image_id;
    if (!fileName && item.image_url) {
      const urlParts = item.image_url.split('/');
      fileName = urlParts[urlParts.length - 1];
    } else if (!fileName) {
      fileName = 'download.jpg'; // Default filename
    }

    try {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();

      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href); // Clean up the object URL
    } catch (error) {
      console.error("Error downloading image:", error);
      // Optionally, add a user notification here
    }
  };

  const closeModal = () => {
    console.log('closeModal called');
    setModalOpened(false);
    setSelectedItem(null); // Clear selected item on close
  };

  if (isSearchPending) {
    console.log('Searching...');
  }
  if (searchError) {
    console.error('Search failed:', searchError.message);
  }

  return (
    <div className='w-full h-full flex flex-col items-center justify-center'>
      {/* Search and Upload UI */}
      <div style={{ width: '80%' }} className='flex items-center gap-2'>
        <TextInput
          style={{ flexGrow: 1 }}
          radius="xl"
          size="lg"
          placeholder="Search images"
          value={inputValue}
          onChange={handleImageSearch}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              executeSearch(inputValue);
            }
          }}
          leftSection={<IconSearch size={18} stroke={1.5} />}
        />
      </div>
      <ActionIcon size="lg" radius="xl" variant="filled" onClick={handleUploadClick} color={theme.primaryColor} style={{ position: 'absolute', right: '1rem', top: '5%', transform: 'translateY(-50%)' }}>
        <IconUpload size={20} stroke={1.5} />
      </ActionIcon>

      {isSearchPending && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
          <Loader />
        </div>
      )}
      {searchError && <p style={{ color: 'red' }}>Error: {searchError.message}</p>}
      
      <Container className='h-full w-full' style={{ padding: '1rem', border: '1px solid #d7d7d7', borderRadius: '1rem', marginTop: '1rem', overflowY: 'scroll' }}>
        {isSearchPending && !searchError ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            {/* Loader shown above */}
          </div>
        ) : searchResult?.length > 0 ? (
          <SimpleGrid 
            cols={{ base: 1, xs: 3 }} 
            spacing="xs" 
            className='h-full'
          >
            {searchResult?.map((item) => {
              return (
                <div 
                  key={item.id} 
                  className='relative cursor-pointer p-2 border border-gray-300 rounded-md hover:shadow-lg transition-shadow'
                  style={{
                    border: '1px solid #f7f4f4', 
                    borderRadius: '1rem', 
                    padding: '0.5rem', 
                    cursor: 'pointer', 
                    position: 'relative' // Added for positioning the score badge
                  }} 
                  onClick={() => handleItemClick(item)} // Make item clickable
                >
                  <ActionIcon
                    variant="subtle" 
                    color="white"
                    size="lg"
                    radius="xl"
                    style={{ position: 'absolute', top: '0.8rem', right: '0.8rem', zIndex: 10, backgroundColor: '#333' }}
                    onClick={(e) => handleDownloadImage(e, item)}
                    title={`Download ${item.name || 'image'}`}
                  >
                    <IconDownload size={20} stroke={1.5} />
                  </ActionIcon>
                  {item.image_url && (
                    <AspectRatio ratio={16 / 9} mb="sm">
                      <Image src={`${import.meta.env.VITE_API_URL}${item.image_url}`} alt={item.name || 'Image preview'} style={{borderRadius: '0.5rem'}}/>
                    </AspectRatio>
                  )}
                  <Text size="sm" fw={500} truncate="end">Name: {item.name || 'Untitled'}</Text>
                  <Text size="xs" c="dimmed">ID: {item.unique_image_id}</Text>
                </div>
              );
            })}
          </SimpleGrid>
        ) : (
          !searchError && <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'gray' }}>No images found</div>
        )}
      </Container>

      {/* Modal for displaying selected item */}
      {selectedItem && (
        <Modal
          opened={modalOpened}
          onClose={closeModal}
          title={selectedItem?.name || "Image Details"}
          size="xl" // Increased size for better layout
          centered
          zIndex={2000}
          keepMounted={false}
        >
          {selectedItem && (
            <Grid>
              <Grid.Col span={{ base: 12, md: 7 }}> {/* Left column for image */}
                {selectedItem.image_url && (
                  <Image
                    src={`${import.meta.env.VITE_API_URL}${encodeURI(selectedItem.image_url)}`}
                    alt={selectedItem.name || 'Selected image'}
                    style={{ width: '100%', height: 'auto', borderRadius: '8px', maxHeight: '70vh', objectFit: 'contain' }}
                  />
                )}
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 5 }} style={{ overflowX: 'scroll' }}> {/* Right column for details */}
                <Stack>
                  
                  {/* Display qdrant_id_uuid if available, otherwise fallback to item.id */}
                  <Text>ID: {selectedItem.unique_image_id || selectedItem.id || 'N/A'}</Text>
                  <Text>Type: {fileType(selectedItem.image_url) || 'N/A'}</Text>
                  {/* <Text>Size: {((selectedItem.fileSize / 1024) / 1024).toFixed(2)} MB</Text> */}
                  <Text>Bildnutzung: {selectedItem.Bildnutzung || 'N/A'}</Text>
                  <Text >Beschreibung: 
                    <div style={{ whiteSpace: 'pre-wrap', height: '20vh', overflowY: 'scroll', fontSize: '14px' }}>
                      
                      {selectedItem.description || 'Untitled'}
                      </div></Text>
                    <Text >AI Beschreibung: 
                  <div style={{ whiteSpace: 'pre-wrap', height: '20vh', overflowY: 'scroll', fontSize: '14px' }}>
                    
                    {selectedItem.gemini_description || 'Untitled'}
                    </div></Text>
                  {/* Display other known fields from the search result item directly */}
                  {selectedItem.caption && typeof selectedItem.caption === 'string' && <Text>Caption: {selectedItem.caption}</Text>}
                  {selectedItem.tags && <Text>Tags: {Array.isArray(selectedItem.tags) ? selectedItem.tags.join(', ') : String(selectedItem.tags)}</Text>}
                  

                  {/* Display other fields from the payload, if payload exists */}
                  {selectedItem.payload && typeof selectedItem.payload === 'object' && (
                    <div style={{ overflowX: 'auto' }}>
                      <Text mt="sm" fw={500}>Additional Details:</Text>
                      {Object.entries(selectedItem.payload).map(([key, value]) => {
                        // Avoid re-displaying fields already shown or too complex/internal
                        const alreadyShownKeys = ['name', 'id', 'qdrant_id_uuid', 'caption', 'tags', 'image_url', 'matched_image_path', 'IMAGE_PATH_COLUMN', 'text_for_search_and_match_for_reranker', 'llm_text_prompt_for_image_description', 'llm_image_description_text'];
                        if (alreadyShownKeys.includes(key) || typeof value === 'object' || value === null || String(value).trim() === '') {
                          return null;
                        }
                        return (
                          <Text size="sm" key={key}>
                            <strong>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> {String(value)}
                          </Text>
                        );
                      })}
                    </div>
                  )}
                </Stack>
              </Grid.Col>
            </Grid>
          )}
        </Modal>
      )}
    </div>
  );
}

export default App;
