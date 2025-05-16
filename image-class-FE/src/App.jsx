import { useState, useEffect } from 'react';
import { IconArrowRight, IconSearch, IconUpload, IconInfoCircle } from '@tabler/icons-react'; 
import { ActionIcon, TextInput, useMantineTheme, Container, Loader, SimpleGrid, Group, Modal, Image, Text, AspectRatio, Grid, Stack, Title } from '@mantine/core'; 
import { notifications } from '@mantine/notifications';
import { useDebouncedCallback } from '@mantine/hooks';
import './App.css';
import { useSearchMutation } from './services/getData';

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

  const debouncedExecuteSearch = useDebouncedCallback(executeSearch, 2000);

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

  console.log('Rendering App component. modalOpened:', modalOpened, 'selectedItem:', selectedItem);

  // Helper function to determine score color and format score display
  const getScoreDisplay = (score) => {
    if (score === undefined || score === null) return null;

    const scorePercent = Math.round(score * 100);
    let color;
    if (score >= 0.85) {
      color = 'green';
    } else if (score >= 0.50) {
      color = 'yellow';
    } else {
      color = 'red';
    }
    return {
      text: `${scorePercent}%`,
      color: color,
    };
  };

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
              const scoreDisplay = getScoreDisplay(item.relevance_score_scaled);
              return (
                <div 
                  key={item.id} 
                  style={{ 
                    border: '1px solid #f7f4f4', 
                    borderRadius: '1rem', 
                    padding: '1rem', 
                    cursor: 'pointer', 
                    position: 'relative' // Added for positioning the score badge
                  }} 
                  onClick={() => handleItemClick(item)} // Make item clickable
                >
                  {scoreDisplay && (
                    <Text 
                      size="xs" 
                      fw={700} 
                      style={{
                        position: 'absolute',
                        top: '0.5rem', 
                        right: '0.5rem',
                        backgroundColor: scoreDisplay.color,
                        color: scoreDisplay.color === 'yellow' ? 'black' : 'white', // Ensure contrast
                        padding: '0.1rem 0.4rem',
                        borderRadius: '0.25rem',
                        zIndex: 1 // Ensure it's above the image
                      }}
                    >
                      {scoreDisplay.text}
                    </Text>
                  )}
                  {item.image_url && (
                    <AspectRatio ratio={16 / 9} mb="sm">
                      <Image src={'http://localhost:8000' + item.image_url} alt={item.name || 'Image preview'} />
                    </AspectRatio>
                  )}
                  <Text size="sm" fw={500} truncate="end">{item.name || 'Untitled'}</Text>
                  <Text size="xs" c="dimmed">ID: {item.id}</Text>
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
                    src={'http://localhost:8000' + selectedItem.image_url}
                    alt={selectedItem.name || 'Selected image'}
                    style={{ width: '100%', height: 'auto', borderRadius: '8px', maxHeight: '70vh', objectFit: 'contain' }}
                  />
                )}
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 5 }}> {/* Right column for details */}
                <Stack>
                  <Title order={3}>{selectedItem.name || 'Untitled'}</Title>
                  
                  {/* Display qdrant_id_uuid if available, otherwise fallback to item.id */}
                  <Text><strong>ID:</strong> {selectedItem.qdrant_id_uuid || selectedItem.id || 'N/A'}</Text>
                  
                  {/* Display other known fields from the search result item directly */}
                  {selectedItem.caption && typeof selectedItem.caption === 'string' && <Text><strong>Caption:</strong> {selectedItem.caption}</Text>}
                  {selectedItem.tags && <Text><strong>Tags:</strong> {Array.isArray(selectedItem.tags) ? selectedItem.tags.join(', ') : String(selectedItem.tags)}</Text>}
                  
                  {selectedItem.relevance_score_scaled !== undefined && (
                    <Text><strong>Score:</strong> {selectedItem.relevance_score_scaled.toFixed(2)}</Text>
                  )}

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
