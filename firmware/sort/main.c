/* Insertion sort: sort an array in-place, return 1 if sorted correctly. */

int main(void) {
    int arr[] = {42, 17, 93, 5, 28, 71, 3, 56, 84, 12};
    int n = 10;

    /* Insertion sort */
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }

    /* Verify: check array is sorted */
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1])
            return 0;  /* fail */
    }

    return 1;  /* pass: array is sorted */
}
