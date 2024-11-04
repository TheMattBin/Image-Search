For RAG using VLM to retrieve metadata

## TODO
- [x] Image search
- [x] Search by query with LLM supported - by metadata
- [ ] Include OCR
- [ ] VectorDB
- [ ] Direct search with vectorDB
- [ ] Integration

## References
- [Reference](https://github.com/gcui-art/album-ai/tree/main)
- [Reference2](https://github.com/hv0905/NekoImageGallery?tab=readme-ov-file)
- [Reference 3](https://medium.com/@myscale/building-a-multi-modal-image-search-application-with-myscale-43b2159e0941)
- [Reference 4](https://towardsdatascience.com/getting-started-with-weaviate-a-beginners-guide-to-search-with-vector-databases-14bbb9285839)

### How to run Docker in Colab

```python
def udocker_init():
    import os
    if not os.path.exists("/home/user"):
        !pip install udocker > /dev/null
        !udocker --allow-root install > /dev/null
        !useradd -m user > /dev/null
    print(f'Docker-in-Colab 1.1.0\n')
    print(f'Usage:     udocker("--help")')
    print(f'Examples:  https://github.com/indigo-dc/udocker?tab=readme-ov-file#examples')

    def execute(command: str):
        user_prompt = "\033[1;32muser@pc\033[0m"
        print(f"{user_prompt}$ udocker {command}")
        !su - user -c "udocker $command"

    return execute

udocker = udocker_init()
```

```shell
!udocker --allow-root images
!udocker --allow-root pull semitechnologies/weaviate:latest
!udocker --allow-root run -p 8080:8080 semitechnologies/weaviate:latest
```


- [udocker](https://github.com/indigo-dc/udocker)
- [udocker-one-off](https://github.com/drengskapur/docker-in-colab)
- [udocker iisue](https://gist.github.com/mwufi/6718b30761cd109f9aff04c5144eb885)