Information Retrieval
---

# Dense Retrieval

Dense information retrieval looks to leverage embeddings as a method for retrieving related documents.

The benefits over sparse methods is the idea of 'semantic' matching. Instead of making a direct match on words where we may have issues with vocabulary mismatch, we can match on the semantic meaning of the query to every document.

However, this leads us to some new issues. How do we know how what a relevant document looks like if we're using
vector distance metrics? How do we efficiently store all the embeddings and then do a inner product search?

We need to consider several problem domains:
- how do we train good embeddings for documents and queries?
  - what data and labels works the best for retrieval models?
  - does the model need ot be jointly trained with documents and queries?
- how do we retrieve candidates from large datasets?
- how do we rank candidates based on embeddings?
- what space tradeoffs can we make for indexing?

# Facebook
Facebook put forward [embedded based search](https://github.com/mtbarta/papers/blob/main/information_retrieval/Embedding-based%20Retrieval%20in%20Facebook%20Search.pdf) where they outline their approach.

They primarily deal with marketplace data. Building on top of FAISS, the authors trained a retrieval model incorporating additional user data.

## Training
FB experiment with several different sources of labels and found that random negative label sampling worked well with clicks as positives. The interesting outcome here was that training with in-batch impressions as negative examples proved to perform worse. The authors hypothesize that these represent harder cases where positive and negative examples are too similar, and the model isn't learning the easy cases as well. 

## Model

FB chose a two tower model where the query and document are jointly trained against a triplet loss. This setup allows the model to be served more efficiently, where the document tower can embed documents offline and the query tower can be used online.

FB found that character trigrams performed well, with word embeddings adding some performance gains at the cost of space. 

## Serving
The data size can quickly beecome impractical to work with. Therefore, there's been a lot of work to find ways to reduce the dimensionality before serving the data. Packages like annoy use trees to partition the data until each partition is small enough. LSH has also been used, but 


FB broke this down into two parts -- coarse quantization and precise quantization.

### Coarse Quantization
Instead of scanning over all full-precision embeddings, we can use an inverse index to store clusters as a coarse quantization. The inverse index can then store a cluster key. We'll compare against each cluster centroid embedding, and then retrieve all full-precision embeddings for those clusters. The authors suggest that clustering parameters become key to good performance, and it's worthwhile fine tuning these parameters.

-- We also need to consider that certain clusters could still be very large. Other approaches to clustering rely on balancing the length of the posting lists.

Coarse quantization takes the form of a k-means algorithm. Even offline, this can be very expensive, and it can be worth using GPUs offline to find the right clustering.

### Product Quantization
Product Quantization is a way to reduce further the amount of data we need to store to do a full inner product search.

[This blog](http://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/) shows some more advanced methods, but
one idea is similar to LSH. We split our vectors, run k-means against all partitions, and use the centroid ids to create
new vectors with a much smaller dimension. We can't calculate the inner product search directly anymore, but we can bookkeep the distance from a vector to the centroid. Summing up all of these partial distances for each partition will give you the total distance from the centroid, and we're able to find the L2 distance between the query vector and each of the centroid vectors.


## RAM vs Disk
Another consideration is where does the data structure live. 


## The Complexity of recall and latency
- our choice of clustering algorithm for dimensionality compression can impact recall dramatically. If a query exists at the boundary of a cluster, we will need to scan more clusters to capture all relevant nearby points.
- This now means that if we change our data, we will need to find a way to update our clusters in real time to avoid
degrading our performance during search.

