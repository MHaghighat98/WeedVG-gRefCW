import torch
import random
from torch.utils.data import Sampler






class UniqueNegSentenceBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains balanced images from all unique negative sentence types.

    Automatically discovers unique negative sentence types in the dataset and balances batches accordingly.
    batch_size must be divisible by the number of unique negative types.

    Example negative types (auto-detected from dataset):
    - Type 1: "no weeds are present in this image" (has crops, no weeds)
    - Type 2: "no crops are present in this image" (has weeds, no crops)
    - Type 3: "no crops or weeds are visible in this image" (empty image)

    Recommended batch_size:
    - For 3 types: 3, 6, 9, 12, 15, 18, 24 (divisible by 3)
    - For N types: batch_size % N == 0
    """

    def __init__(self, image_indices, image_idx_to_neg_sentence, batch_size=3, drop_last=False):
        self.image_indices = list(image_indices)
        self.image_idx_to_neg_sentence = image_idx_to_neg_sentence
        self.batch_size = batch_size
        self.drop_last = drop_last

        # CRITICAL FIX: Remove sentence normalization to preserve distinct sentence types
        # Normalization was causing different sentence types to be grouped together incorrectly

        # CRITICAL FIX: Separate images with and without negative sentences
        # Images with empty negatives (multi-class) can be mixed into any batch
        self.images_without_negatives = [
            idx for idx in self.image_indices if self.image_idx_to_neg_sentence[idx] is None or self.image_idx_to_neg_sentence[idx] == ""
        ]

        # Automatically discover unique negative sentence types (excluding empty)
        unique_neg_sentences = set(
            sent
            for sent in self.image_idx_to_neg_sentence.values()
            if sent is not None and sent != ""  # Filter out None/empty
        )

        # USER REQUEST: Treat empty negatives as a fourth balanced type
        # This ensures all four types are represented in each batch
        if self.images_without_negatives:
            # Add empty negatives as a fourth "negative type"
            unique_neg_sentences.add("")  # Empty string represents empty negatives

        if not unique_neg_sentences:
            # EDGE CASE: No images with negative sentences at all
            # This means all images are multi-class - just use random batching
            import warnings

            warnings.warn(
                "No negative sentences found in dataset! All images appear to be multi-class. "
                "Using simple random batching instead of balanced sampling.",
                UserWarning,
            )
            self.neg_sentence_types = []
            self.num_types = 0
            self.images_per_type = 0
            self.image_groups = {}
            self.simple_batching = True
            return

        self.simple_batching = False
        self.neg_sentence_types = sorted(list(unique_neg_sentences))  # Deterministic order
        self.num_types = len(self.neg_sentence_types)

        # CRITICAL FIX: Make batch_size flexible - automatically adjust images_per_type
        # Instead of requiring batch_size % num_types == 0, adapt to what's possible
        if batch_size < self.num_types:
            import warnings

            warnings.warn(
                f"batch_size ({batch_size}) is smaller than number of unique negative types ({self.num_types}). "
                f"Setting batch_size={self.num_types} for representation guarantee.",
                UserWarning,
            )
            self.batch_size = self.num_types
        # No longer require batch_size % num_types == 0 - we fill flexibly

        self.images_per_type = 1  # Minimum per type (for legacy compatibility)

        # CRITICAL FIX: Group images using original sentences (no normalization)
        self.image_groups = {neg_sent: [] for neg_sent in self.neg_sentence_types}
        for idx in self.image_indices:
            neg_sent = self.image_idx_to_neg_sentence[idx]

            # Handle empty negatives
            if neg_sent is None or neg_sent == "":
                if "" in self.image_groups:
                    self.image_groups[""].append(idx)
                continue

            if neg_sent in self.image_groups:
                self.image_groups[neg_sent].append(idx)
            else:
                # This should not happen
                import warnings

                warnings.warn(
                    f"Image index {idx} has sentence '{neg_sent}' not found in neg_sentence_types: {self.neg_sentence_types}",
                    UserWarning,
                )

        # Clear images_without_negatives since they're now in image_groups[""]
        self.images_without_negatives = []

        # Print summary
        print("=> UniqueNegSentenceBatchSampler initialized:")
        print(f"   Batch size: {self.batch_size} (maximizing data usage with flexible type distribution)")
        print(f"   Discovered {self.num_types} unique negative sentence types:")
        for i, neg_sent in enumerate(self.neg_sentence_types, 1):
            count = len(self.image_groups[neg_sent])
            if neg_sent == "":
                print(f"   Type {i}: '(empty negatives)' ({count} images)")
            else:
                print(f"   Type {i}: '{neg_sent}' ({count} images)")
        if self.images_without_negatives:
            print(f"   Images without negatives (multi-class): {len(self.images_without_negatives)}")
            print("   (These will be mixed into batches for positive-only training)")
        print(f"   Total images: {len(self.image_indices)}")
        print(f"   Expected batches: {len(self)} (using all available training data)")


    def __iter__(self):
        """
        Generate batches ensuring ALL batches contain all four negative sentence types
        while using ALL available training images through cycling of exhausted types.

        Strategy:
        - Each batch gets exactly 1 image from each of the 4 types (guaranteed representation)
        - When a type is exhausted, cycle back to the beginning with reshuffling
        - Fill remaining slots with any available images (cycling exhausted types as needed)
        - Uses 100% of training data while maintaining type balance in every batch

        This achieves both full data utilization and guaranteed type diversity.
        """
        # Shuffle each group independently
        shuffled_groups = {}
        for neg_sent in self.neg_sentence_types:
            group = self.image_groups[neg_sent].copy()
            random.shuffle(group)
            shuffled_groups[neg_sent] = group

        # Shuffle images without negatives separately (if any remain)
        shuffled_no_neg = self.images_without_negatives.copy()
        random.shuffle(shuffled_no_neg)

        batch_idx = 0
        total_images_used = 0
        used_images = set()  # Track all used images to ensure each is used exactly once

        while total_images_used < len(self.image_indices):
            batch = []

            # Step 1: Take exactly 1 from each type (guaranteed, with cycling for exhausted types)
            for neg_sent in self.neg_sentence_types:
                group = self.image_groups[neg_sent]  # Use original groups, not shuffled ones for cycling
                available_images = [i for i in group if i not in used_images]

                if available_images:
                    # Take a random unused image from this type
                    image_idx = random.choice(available_images)
                    batch.append(image_idx)
                    used_images.add(image_idx)
                    total_images_used += 1
                else:
                    # All images of this type have been used - cycle back and reuse
                    # This ensures every batch has all types, even if it means reusing images
                    image_idx = random.choice(group)
                    batch.append(image_idx)
                    # Don't add to used_images since we're reusing

            # Step 2: Fill remaining slots with unused images from any types
            remaining_slots = self.batch_size - len(batch)
            filled_slots = 0

            # Collect all unused images from all types
            all_unused_images = []
            for neg_sent in self.neg_sentence_types:
                all_unused_images.extend([i for i in self.image_groups[neg_sent] if i not in used_images])

            # Add unused supplementary images
            all_unused_images.extend([i for i in self.images_without_negatives if i not in used_images])

            # Randomly shuffle all unused images for filling
            random.shuffle(all_unused_images)

            # Fill the batch with unused images
            for image_idx in all_unused_images:
                if filled_slots >= remaining_slots:
                    break
                batch.append(image_idx)
                used_images.add(image_idx)
                total_images_used += 1
                filled_slots += 1

            # If we still need more slots and have used all unused images, fill with any images (allowing some reuse)
            if filled_slots < remaining_slots:
                # Get all images for filling (allowing reuse of already used images for filling only)
                all_images_for_filling = []
                for neg_sent in self.neg_sentence_types:
                    all_images_for_filling.extend(self.image_groups[neg_sent])
                all_images_for_filling.extend(self.images_without_negatives)

                # Remove images already in this batch to avoid immediate duplicates
                batch_set = set(batch)
                fill_candidates = [i for i in all_images_for_filling if i not in batch_set]

                random.shuffle(fill_candidates)

                for image_idx in fill_candidates:
                    if filled_slots >= remaining_slots:
                        break
                    batch.append(image_idx)
                    filled_slots += 1

            # Yield the batch
            if len(batch) > 0:
                # Shuffle within batch to avoid type ordering bias
                random.shuffle(batch)
                yield batch
                batch_idx += 1

                # Safety check: if we've used all unique images, stop
                if len(used_images) >= len(self.image_indices):
                    break

    def __len__(self):
        """Calculate total number of batches using all images with guaranteed type representation."""
        total_images = sum(len(group) for group in self.image_groups.values()) + len(self.images_without_negatives)
        return (total_images + self.batch_size - 1) // self.batch_size
