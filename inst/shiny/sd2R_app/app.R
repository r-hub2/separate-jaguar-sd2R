# sd2R Shiny GUI — text-to-image generation
# Launch via sd2R::sd_app() or sd2R::sd_app(model_dir = "/path/to/models")

library(shiny)

# ---------- Model presets by architecture ----------
MODEL_PRESETS <- list(
  sd1 = list(
    label = "SD 1.x",
    width = 512L, height = 512L,
    steps = 20L, cfg = 7.0,
    sampler = "EULER_A", scheduler = "KARRAS",
    max_chars = 350,
    resolutions = c("512x512", "512x768", "768x512", "768x768")
  ),
  sd2 = list(
    label = "SD 2.x",
    width = 768L, height = 768L,
    steps = 20L, cfg = 7.0,
    sampler = "EULER_A", scheduler = "KARRAS",
    max_chars = 350,
    resolutions = c("512x512", "512x768", "768x512", "768x768")
  ),
  sdxl = list(
    label = "SDXL",
    width = 1024L, height = 1024L,
    steps = 25L, cfg = 5.0,
    sampler = "EULER", scheduler = "KARRAS",
    max_chars = 700,
    resolutions = c("1024x1024", "1152x896", "896x1152",
                    "1216x832", "832x1216", "768x768")
  ),
  flux = list(
    label = "Flux",
    width = 1024L, height = 1024L,
    steps = 20L, cfg = 1.0,
    sampler = "EULER", scheduler = "SIMPLE",
    max_chars = 2000,
    resolutions = c("1024x1024", "1152x896", "896x1152",
                    "1216x832", "832x1216", "768x1360", "1360x768")
  ),
  sd3 = list(
    label = "SD 3",
    width = 1024L, height = 1024L,
    steps = 28L, cfg = 5.0,
    sampler = "EULER", scheduler = "SGM_UNIFORM",
    max_chars = 700,
    resolutions = c("1024x1024", "1152x896", "896x1152",
                    "1216x832", "832x1216")
  )
)

sampler_names  <- names(sd2R::SAMPLE_METHOD)
scheduler_names <- names(sd2R::SCHEDULER)

# ---------- Auto-assign model roles by filename ----------
auto_assign_roles <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.(safetensors|gguf|ckpt)$",
                      full.names = FALSE, ignore.case = TRUE)
  if (length(files) == 0) return(list())

  sizes <- file.size(file.path(dir_path, files))
  names(sizes) <- files
  fl <- tolower(files)

  roles <- list(model = "", diffusion = "", vae = "", clip_l = "", t5xxl = "")
  assigned <- rep(FALSE, length(files))

  # VAE: "vae" or standalone "ae" in name
  idx <- grep("(^|[^a-z])(vae|\\bae\\b)", fl)
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$vae <- files[pick]
    assigned[pick] <- TRUE
  }

  # CLIP-L: "clip" in name
  idx <- grep("clip", fl)
  idx <- setdiff(idx, which(assigned))
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$clip_l <- files[pick]
    assigned[pick] <- TRUE
  }

  # T5-XXL: "t5" in name
  idx <- grep("t5", fl)
  idx <- setdiff(idx, which(assigned))
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$t5xxl <- files[pick]
    assigned[pick] <- TRUE
  }

  # Diffusion: "flux", "sd3", "dit", "unet" in name — separate diffusion model
  idx <- grep("flux|sd3|dit|unet", fl)
  idx <- setdiff(idx, which(assigned))
  if (length(idx)) {
    pick <- idx[which.max(sizes[idx])]
    roles$diffusion <- files[pick]
    assigned[pick] <- TRUE
  }

  # Remaining: largest unassigned = main model (single-file checkpoints like SD1/SDXL)
  remaining <- which(!assigned)
  if (length(remaining)) {
    pick <- remaining[which.max(sizes[remaining])]
    roles$model <- files[pick]
  }

  roles
}

# Read initial model_dir from option set by sd_app()
init_model_dir <- getOption("sd2R.model_dir", default = "")

# ---------- UI ----------
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    .well { background: #16213e; border: 1px solid #2a3a5c; }
    .btn-primary { background: #0f3460; border-color: #1a5276; }
    .btn-primary:hover { background: #1a5276; }
    .btn-danger { background: #c0392b; border-color: #a93226; }
    .form-control, .selectize-input { background: #0f3460; color: #e0e0e0;
      border-color: #2a3a5c; }
    .selectize-dropdown { background: #16213e; color: #e0e0e0; }
    .selectize-dropdown-content .option.active { background: #1a5276; }
    h3, h4 { color: #e94560; }
    .progress { background: #0f3460; }
    .progress-bar { background: #e94560; }
    #gpu_info { font-family: monospace; font-size: 0.85em; white-space: pre-wrap;
      background: #0f3460; padding: 8px; border-radius: 4px; margin-bottom: 10px; }
    #char_counter { font-size: 0.85em; margin-top: -8px; margin-bottom: 8px; }
    .img-container { text-align: center; padding: 10px; }
    .img-container img { max-width: 100%; border: 2px solid #2a3a5c; border-radius: 4px; }
    #status_text { font-style: italic; color: #aaa; }
  "))),

  titlePanel(
    div(
      span("sd2R", style = "color:#e94560; font-weight:bold;"),
      span(" Image Generator", style = "color:#e0e0e0;")
    )
  ),

  sidebarLayout(
    sidebarPanel(
      width = 4,

      # GPU info
      h4("GPU"),
      uiOutput("gpu_info"),

      # Model
      h4("Model"),
      selectInput("model_type", "Architecture", names(MODEL_PRESETS),
                  selected = "sd1"),

      # Models folder
      fluidRow(
        column(9, textInput("model_dir", "Models folder", value = init_model_dir)),
        column(3, actionButton("scan_dir", "Scan", class = "btn-primary btn-sm",
                               style = "margin-top: 25px; width: 100%;"))
      ),

      # Auto-assigned dropdowns
      selectInput("sel_model", "Model", choices = NULL),
      selectInput("sel_vae", "VAE (optional)", choices = NULL),
      conditionalPanel(
        condition = "input.model_type == 'flux' || input.model_type == 'sd3'",
        selectInput("sel_diffusion", "Diffusion model", choices = NULL),
        selectInput("sel_clip_l", "CLIP-L (optional)", choices = NULL),
        selectInput("sel_t5xxl", "T5-XXL (optional)", choices = NULL)
      ),

      actionButton("load_model", "Load Model", class = "btn-primary btn-block",
                    style = "width: 100%; margin-bottom: 15px;"),

      hr(),

      # Generation params
      h4("Generation"),

      textAreaInput("prompt", "Prompt", rows = 4, placeholder = "Describe your image..."),
      uiOutput("char_counter"),
      textAreaInput("neg_prompt", "Negative prompt", rows = 2,
                    value = "bad quality, blurry, ugly"),

      selectInput("resolution", "Resolution", choices = NULL),
      fluidRow(
        column(6, selectInput("sampler", "Sampler", sampler_names, selected = "EULER_A")),
        column(6, selectInput("scheduler", "Scheduler", scheduler_names, selected = "KARRAS"))
      ),
      fluidRow(
        column(4, numericInput("steps", "Steps", 20, min = 1, max = 100)),
        column(4, numericInput("cfg", "CFG", 7.0, min = 0, max = 30, step = 0.5)),
        column(4, numericInput("seed", "Seed", 42, min = -1))
      ),

      hr(),
      fluidRow(
        column(6,
          actionButton("generate", "Generate", class = "btn-primary btn-block",
                       style = "width: 100%;")
        ),
        column(6,
          downloadButton("save_btn", "Save PNG", class = "btn-block",
                         style = "width: 100%;")
        )
      )
    ),

    mainPanel(
      width = 8,
      textOutput("status_text"),
      div(class = "img-container", uiOutput("result_image"))
    )
  )
)

# ---------- Server ----------
server <- function(input, output, session) {

  rv <- reactiveValues(
    ctx = NULL,
    last_image = NULL
  )

  # GPU info at startup
  output$gpu_info <- renderUI({
    info <- tryCatch({
      if (!sd2R::sd_vulkan_device_count()) {
        "No Vulkan GPU detected"
      } else {
        devs <- ggmlR::ggml_vulkan_list_devices()
        lines <- vapply(devs, function(d) {
          sprintf("[%d] %s  (%.1f / %.1f GB)",
                  d$index, d$name,
                  d$free_memory / 1e9, d$total_memory / 1e9)
        }, character(1))
        paste(lines, collapse = "\n")
      }
    }, error = function(e) paste("GPU info error:", e$message))
    div(id = "gpu_info", info)
  })

  # --- Scan folder: list files, auto-assign roles, populate dropdowns ---
  scan_model_dir <- function() {
    dir_path <- trimws(input$model_dir)
    if (!nzchar(dir_path) || !dir.exists(dir_path)) {
      showNotification("Folder not found", type = "error")
      return()
    }

    all_files <- list.files(dir_path,
                            pattern = "\\.(safetensors|gguf|ckpt)$",
                            full.names = FALSE, ignore.case = TRUE)
    if (length(all_files) == 0) {
      showNotification("No model files found in folder", type = "warning")
      return()
    }

    none <- c("(none)" = "")
    choices     <- setNames(all_files, all_files)
    choices_opt <- c(none, choices)

    roles <- auto_assign_roles(dir_path)

    updateSelectInput(session, "sel_model",     choices = choices_opt, selected = roles$model)
    updateSelectInput(session, "sel_diffusion", choices = choices_opt, selected = roles$diffusion)
    updateSelectInput(session, "sel_vae",       choices = choices_opt, selected = roles$vae)
    updateSelectInput(session, "sel_clip_l",    choices = choices_opt, selected = roles$clip_l)
    updateSelectInput(session, "sel_t5xxl",     choices = choices_opt, selected = roles$t5xxl)

    showNotification(sprintf("Found %d model files", length(all_files)), type = "message")
  }

  # Scan on button click
  observeEvent(input$scan_dir, scan_model_dir())

  # Auto-scan if model_dir was passed via sd_app()
  if (nzchar(init_model_dir) && dir.exists(init_model_dir)) {
    observe({
      scan_model_dir()
    }, once = TRUE)
  }

  # --- Resolve model paths ---
  get_model_paths <- function() {
    dir_path <- trimws(input$model_dir)
    if (!nzchar(dir_path)) return(list())
    full <- function(f) {
      if (is.null(f) || !nzchar(f)) return(NULL)
      file.path(dir_path, f)
    }
    list(
      model_path           = full(input$sel_model),
      diffusion_model_path = full(input$sel_diffusion),
      vae_path             = full(input$sel_vae),
      clip_l_path          = full(input$sel_clip_l),
      t5xxl_path           = full(input$sel_t5xxl)
    )
  }

  # Update controls when preset changes
  observeEvent(input$model_type, {
    p <- MODEL_PRESETS[[input$model_type]]
    updateSelectInput(session, "resolution", choices = p$resolutions,
                      selected = paste0(p$width, "x", p$height))
    updateSelectInput(session, "sampler", selected = p$sampler)
    updateSelectInput(session, "scheduler", selected = p$scheduler)
    updateNumericInput(session, "steps", value = p$steps)
    updateNumericInput(session, "cfg", value = p$cfg)
  })

  # Char counter
  output$char_counter <- renderUI({
    p <- MODEL_PRESETS[[input$model_type]]
    n <- nchar(input$prompt %||% "")
    color <- if (n > p$max_chars) "#e94560" else "#888"
    div(id = "char_counter",
        span(sprintf("%d / %d characters", n, p$max_chars), style = paste0("color:", color)))
  })

  # Load model
  observeEvent(input$load_model, {
    paths <- get_model_paths()

    if (is.null(paths$model_path) && is.null(paths$diffusion_model_path)) {
      showNotification("Select a model or diffusion model file", type = "error")
      return()
    }

    output$status_text <- renderText("Loading model...")

    tryCatch({
      args <- list(model_type = input$model_type, verbose = TRUE)
      if (!is.null(paths$model_path))
        args$model_path <- paths$model_path
      if (!is.null(paths$diffusion_model_path))
        args$diffusion_model_path <- paths$diffusion_model_path
      if (!is.null(paths$vae_path))
        args$vae_path <- paths$vae_path
      if (!is.null(paths$clip_l_path))
        args$clip_l_path <- paths$clip_l_path
      if (!is.null(paths$t5xxl_path))
        args$t5xxl_path <- paths$t5xxl_path

      rv$ctx <- do.call(sd2R::sd_ctx, args)
      output$status_text <- renderText("Model loaded.")
      showNotification("Model loaded successfully", type = "message")
    }, error = function(e) {
      output$status_text <- renderText(paste("Load error:", e$message))
      showNotification(e$message, type = "error")
    })
  })

  # Generate
  observeEvent(input$generate, {
    if (is.null(rv$ctx)) {
      showNotification("Load a model first", type = "error")
      return()
    }
    if (!nzchar(input$prompt %||% "")) {
      showNotification("Enter a prompt", type = "error")
      return()
    }

    dims <- as.integer(strsplit(input$resolution, "x")[[1]])
    output$status_text <- renderText("Generating...")

    tryCatch({
      withProgress(message = "Generating image", value = 0, {
        imgs <- sd2R::sd_txt2img(
          rv$ctx,
          prompt = input$prompt,
          negative_prompt = input$neg_prompt %||% "",
          width = dims[1],
          height = dims[2],
          sample_method = sd2R::SAMPLE_METHOD[[input$sampler]],
          sample_steps = as.integer(input$steps),
          cfg_scale = as.numeric(input$cfg),
          seed = as.integer(input$seed),
          scheduler = sd2R::SCHEDULER[[input$scheduler]],
          batch_count = 1L
        )
        setProgress(1)
      })

      rv$last_image <- imgs[[1]]
      output$status_text <- renderText(
        sprintf("Done. %dx%d, seed=%d", dims[1], dims[2], input$seed))
    }, error = function(e) {
      output$status_text <- renderText(paste("Error:", e$message))
      showNotification(e$message, type = "error", duration = 10)
    })
  })

  # Display result
  output$result_image <- renderUI({
    img <- rv$last_image
    if (is.null(img)) {
      div(style = "color:#555; padding: 100px 0; font-size: 1.3em;",
          "Generated image will appear here")
    } else {
      tmp <- tempfile(fileext = ".png")
      sd2R::sd_save_image(img, tmp)
      b64 <- base64enc::base64encode(tmp)
      tags$img(src = paste0("data:image/png;base64,", b64),
               style = "max-width: 100%;")
    }
  })

  # Download
  output$save_btn <- downloadHandler(
    filename = function() {
      paste0("sd2R_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".png")
    },
    content = function(file) {
      if (!is.null(rv$last_image)) {
        sd2R::sd_save_image(rv$last_image, file)
      }
    }
  )
}

shinyApp(ui, server)
