classdef hjb_grid < rl.env.MATLABEnvironment
    
    
    %% Properties
    properties (Constant)
        param = struct(...
            ... %% PDE parameters
            'Target', [3; 0], ... % Target state
            'IC', [-2.5; 0], ... % initial state
            'xdims', 2, ... % dimensions of state space
            'udims', 1, ... % dimensions of control space
            'h', 0.01, ... % dynamics integration time-step            
            ... %% Grid parameters
            'num_points', 1500, ... % initial number of nodes for the QMC grid
            'du', 150, ... % control grid resolution
            'bd', [-6 6; -6 6], ... % computation domain boundary
            'marg', 0, ... % margin area around the boundary
             ... %% Node generation parameters
            'duplicate_tol', 1E-10, ... % distance between nodes to be considered duplicates
            'skip_QMC', 2E3, ... % number of QMC's Sobol sequence terms to skip (for better uniform properties)
            'leap_QMC', 1e2, ... % number of points in the sequence to leap over and omit for every point taken
            ... %% Initial solution solver parameters
            'initial_guess_V', 0.7, ... % initial transformed value function guess
            'tol_fp', 1E-14, ... % solution convergence threshold
            'load_solution', true, ... % true=load initial guess data
            ... %% Computation parameters
            'chunk_size', 100, ... % number of nodes per vectorized chunk for parallel computation
            'error_chunk_size', 1000, ... % number of nodes per vectorized chunk for residual parallel computation
            'num_samples', 10000, ... % number of integration samples            
            ... %% Training parameters
            'min_nodes', 0.3 * 1500, ... % minimum node population for every episode
            'max_nodes', 1.5 * 1500, ... % maximum node count for every episode
            'max_steps', 32, ... % maximum number of steps per training episode           
            ... %% Action parameters            
            'minR', 0, ... % minimum percentage of domain radius for node generation area
            'maxR', 0.45, ... % maximum percentage of domain radius for node generation area
            'minP', -1, ... % minimum node population percent change
            'maxP', 1, ... % maximum node population percent change
            'action_penalty', true, ... % true=penalize out of bounds action values
            ... %% Display parameters (display toggles)
            'iter_info_all', false, ... % true=display every HJB iteration information
            'reward_info', false, ... % true=display reward details every step
            'terminal_reward', false, ... % true=display total reward at the last step
            'env_info',false, ... % true=display state & action choices every step
            'plot_grid', false, ... % true=plot the grid during training
            'plot_traj', false, ... % true=plot the system trajectories on the grid
            'nodes_surface', false, ... % true=plot nodes on the error contour
            'plot_surface', false ... % true=plot the solution surface dynamically
        );
    end

    properties
        n; % current node count
        x; % current grid node coordinates 
        V; % current value functions on x       
        u; % current optimal control on x       
        I; % current interpolation model for value function
        Iu; % interpolation model for control
        error_norm; % integrated error norm for residual error over the domain        
        n_init; % initial node count
        x_init; % initial grid coordinates
        V_init; % initial value functions on x_init       
        u_init; % initial optimal control on x_init        
        I_init; % initial interpolation model on x_init       
        Iu_init; % initial control interpolation model on x_init   
        error_norm_init; % initial integrated error norm reference value
        errorV; % error value at the sampled node
        density; % local density at the sampled node
        dist_ratio; % shortest distance from the sampled node to the boundary
        domain_R; % relative domain radius
        point; % currently sampled point for local state values
        dist; % distances from all nodes to the sampled point
        steps; % current training step count
        plot_grid; % handle for grid plot
        plot_surface; % handle for value function plot
        nodes_surface; % handle for nodes and error contour plot
    end

    %% Methods
    methods
        %%%% Contructor method creates an instance of the environment: hjb_grid
        function this = hjb_grid()
            % Get initial parameters before superclass constructor setup
            param = hjb_grid.param; 
            
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([3 1]);
            ObservationInfo.Name = 'hjb_grid State';
            ObservationInfo.Description = 'local error, local density, boundary proximity ratio';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([2 1]);
            ActionInfo.Name = 'hjb_grid Action';
            ActionInfo.Description = 'percentage of domain radius, percentage of node population';
            ActionInfo.LowerLimit = [param.minR; param.minP];
            ActionInfo.UpperLimit = [param.maxR; param.maxP];
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            if ~param.load_solution
                % Get 'x' and 'n' for initial grid
                [this.x_init, this.n_init] = initial_grid(this.param);
            
                % Compute PDE (iterated) solution on the initial grid
                [this.V_init, this.u_init, this.I_init, this.Iu_init] = compute_solution(this.x_init, this.n_init, this.param);
            else
                % Load the precomputed guess file
                fprintf('Loading precomputed solution...\n');
                data = load('precomputed_solution.mat', 'x', 'n', 'V', 'u', 'I', 'Iu');
                this.x_init = data.x; this.n_init = data.n;
                this.V_init = data.V; this.u_init = data.u; 
                this.I_init = data.I; this.Iu_init = data.Iu;
                fprintf('Precomputed solution successfully loaded.\n');
            end
            
            % Reference integrated error
            [this.error_norm_init] = residual_integral(this.I_init, this.Iu_init, this.param);
            
            % Compute the relative domain radius
            [this.domain_R] = domain_radius(this.param);
            
            % Initialize the plots
            this.plot_grid = update_plot(this.x_init, this.error_norm_init, this.I_init, this.Iu_init, [], this.plot_grid, this.param); 
            this.plot_surface = update_surface(this.x_init, this.V_init, this.plot_surface, this.param);
            this.nodes_surface = update_contour(this.x_init, this.I_init, this.Iu_init, this.error_norm_init, this.nodes_surface, this.param);
        end
        
        %%%% Step function
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            
            % Update steps
            this.steps = this.steps + 1;
            
            % Notify user
            if this.param.env_info
                fprintf('Environment Step Information:\n');
                fprintf('====================================================================\n');
                fprintf('| %-30s | %-25s  \n', 'Metric', 'Value');
                fprintf('--------------------------------------------------------------------\n');
                fprintf('| %-30s | %d                         \n', 'Step', this.steps);
                fprintf('| %-30s | (%.4f, %.4f)               \n', 'Sampled Point (x, y)', this.point(1), this.point(2));
                fprintf('| %-30s | %.4f                       \n', 'Action: Percent Domain Radius', Action(1));
                fprintf('| %-30s | %.4f                       \n', 'Action: Percent Node Change', Action(2));                
                fprintf('| %-30s | %.4f                       \n', 'State: Node Error', this.errorV);
                fprintf('| %-30s | %.4f                       \n', 'State: Node Density', this.density);
                fprintf('| %-30s | %.4f                       \n', 'State: Boundary Distance Ratio', this.dist_ratio);
                fprintf('====================================================================\n\n');
            end                        
            
            % Clip action (PPO is probabilistic)
            [R, P] = prepare_action(Action, this.domain_R, this.param);
            
            %%% Apply action
            [this.x, this.n, this.V, this.u, this.I, this.Iu] = ...
            apply_action(R, P, this.point, this.dist, this.x, this.n, this.V, this.u, this.I, this.Iu, this.param);
            
            % Integrate residual over the domain
            [this.error_norm] = residual_integral(this.I, this.Iu, this.param);
            
            %%% Calculate reward
            [Reward, IsDone] = ...
            compute_reward(this.error_norm, this.error_norm_init, this.n, this.n_init, this.steps, Action, this.param);
            
            %%% Prepare the observation
            % Sample a point
            [this.point] = sample_point(this.param);
            % Get node density at the sampled point
            [this.density, this.dist] = knn_density(this.point, this.x, this.n, this.param);
            % Compute local error at the sampled point
            [this.errorV] = compute_error(this.point, this.I, this.Iu, this.param);
            % Get minimum to maximum boundary distance ratio
            this.dist_ratio = distance_ratio(this.point, this.param);
            % Prepare the next state
            Observation = [this.errorV; this.density; this.dist_ratio];                      
            
            % Update plot 
            this.plot_grid = update_plot(this.x, this.error_norm, this.I, this.Iu, IsDone, this.plot_grid, this.param);
            this.plot_surface = update_surface(this.x, this.V, this.plot_surface, this.param);
            this.nodes_surface = update_contour(this.x, this.I, this.Iu, this.error_norm, this.nodes_surface, this.param);
            
            % Notify that the environment has been updated
            notifyEnvUpdated(this);
        end
        
        %%%% Reset function
        function InitialObservation = reset(this)
        
            % Reset steps
            this.steps = 0;
            
            % Assign initial values for the current episode
            this.x = this.x_init; this.n = this.n_init;
            this.V = this.V_init; this.u = this.u_init;
            this.I = this.I_init; this.Iu = this.Iu_init;
            this.error_norm = this.error_norm_init;
            
            %%% Prepare the initial observation
            % Sample a point
            [this.point] = sample_point(this.param);
            % Get node density at the sampled point
            [this.density, this.dist] = knn_density(this.point, this.x, this.n, this.param);
            % Compute local error at the sampled point
            [this.errorV] = compute_error(this.point, this.I, this.Iu, this.param);
            % Get minimum to maximum boundary distance ratio
            this.dist_ratio = distance_ratio(this.point, this.param);
            % Agent's (initial) state
            InitialObservation = [this.errorV; this.density; this.dist_ratio];
            
            % Notify that the environment has been updated
            notifyEnvUpdated(this);
        end
    end    
end