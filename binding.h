#ifdef __cplusplus
#include <vector>
#include <string>
extern "C"
{
#endif

#include <stdbool.h>

    extern unsigned char tokenCallback(void *, const char *);

    int load_state(void *ctx, char *statefile, char *modes);

    int eval(void *params_ptr, void *ctx, char *text);

    void save_state(void *ctx, char *dst, char *modes);

    // contains the created context and the model returned from load_model()
    typedef struct load_model_result {
        void* model;
        void* ctx;
    } load_model_result;
    load_model_result load_model(const char *fname, int n_ctx, int n_seed, bool mlock, bool embeddings, bool mmap, bool vocab_only, 
                                int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit, bool numa, float rope_freq, float rope_scale);

    int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings, int *res_n_embeddings);

    int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize, float *res_embeddings, int *res_n_embeddings);

    // returns the size of the embeddings this model makes
    int get_llama_n_embd(void *model);

    void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k, float top_p, float min_p, float temp, float repeat_penalty,
                                int repeat_last_n, bool ignore_eos, int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                                float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, 
                                float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, 
                                bool mmap, const char *maingpu, const char *tensorsplit, bool prompt_cache_ro, float rope_freq_base, float rope_freq_scale, 
                                int n_draft);

    void llama_free_params(void *params_ptr);

    void llama_binding_free_model(void *state, void *model_ptr);

    // performance timing information
    typedef struct llama_predict_result {
        // 0 == success; 1 == failure
        int result;
        
        // timing data
        double t_start_ms;
        double t_end_ms;
        double t_load_ms;
        double t_sample_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int n_sample;
        int n_p_eval;
        int n_eval;
    } llama_predict_result;

    llama_predict_result llama_predict(void *params_ptr, void *ctx_ptr, void *model_ptr, char *result, bool debug);    

#ifdef __cplusplus
}

std::vector<std::string> create_vector(const char **strings, int count);
void delete_vector(std::vector<std::string> *vec);
#endif