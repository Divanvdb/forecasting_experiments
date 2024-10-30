import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML

import cartopy.crs as ccrs

import torch
from torch.utils.data import Dataset, DataLoader

from typing import Tuple

# Add settings to choose variables and spatial extent

class WeatherData(Dataset):
    def __init__(self, 
                 window_size: int = 24, 
                 step_size: int = 12, 
                 set:str = 'train', 
                 area: Tuple[int, int] = (-31.667, 18.239), 
                 spaces: int = 0, 
                 intervals: int = 1,
                 lightning: bool = False,
                 series_target: bool = False,
                 verbose: bool = False):
        '''

        Data format:
            0 - humidity
            1 - temperature
            2 - u-wind
            3 - v-wind
            4 - w-wind

        '''
        print('10%', end='\r')

        # Extract correct dataset
        if set == 'train':
            years = ['2018', '2019', '2020', '2021']
        elif set == 'val':
            years = ['2022']
        elif set == 'test':
            years = ['2023']

        if lightning:

            self.data = np.concatenate([np.load(f'forecasting_experiments/datasets/{year}_850_SA.npy') for year in years], axis=1)
            self.data = self.data.transpose(1, 2, 3, 0)

            self.times = np.concatenate([np.load(f'forecasting_experiments/datasets/{year}_850_SA_times.npy') for year in years])

            self.lon = np.load('forecasting_experiments/datasets/SA_lon.npy')
            self.lat = np.load('forecasting_experiments/datasets/SA_lat.npy')

        else:
            self.data = np.concatenate([np.load(f'datasets/{year}_850_SA.npy') for year in years], axis=1)
            self.data = self.data.transpose(1, 2, 3, 0)

            self.times = np.concatenate([np.load(f'datasets/{year}_850_SA_times.npy') for year in years])

            self.lon = np.load('datasets/SA_lon.npy')
            self.lat = np.load('datasets/SA_lat.npy')

        print('40%', end='\r')

        # Get lat and long

        self.spaces = spaces

        self.get_area(area)

        print('50%', end='\r')

        # Normalize data and sort into variables

        if spaces != 0:
            q = self.data[:, :, :, 0]
            t = self.data[:, :, :, 1]
            u = self.data[:, :, :, 2]
            v = self.data[:, :, :, 3]
            w = self.data[:, :, :, 4]
        else:
            q = self.data[:,0]
            t = self.data[:,1]
            u = self.data[:,2]
            v = self.data[:,3]
            w = self.data[:,4]

        q, t, u, v, w = self.normalize(q, t, u, v, w)

        print('70%', end='\r')

        # Calculate wind speed and direction
        self.series_target = series_target

        self.calculate_wind(u, v)

        print('90%', end='\r')

        # Serup the dataloader
        if self.series_target:
            self.features = torch.tensor(np.stack([q, t, u, v, w], axis=-1), dtype=torch.float32)
        else:
            self.features = torch.tensor(np.stack([q, t, u, v, w, self.wspd], axis=-1), dtype=torch.float32)

        self.targets = torch.tensor(self.wspd, dtype=torch.float32)
        self.window_size = window_size
        self.step_size = step_size

        print('100%', end='\r')

        self.intervals = intervals

        if verbose:
            print(f'Details for {set} set:')

            print(f'Data from {years} loaded')
            
            print(f'Features shape: {self.features.shape}')
            print(f'Targets shape: {self.targets.shape}')

            print(f'Longitudes: {self.lon}')
            print(f'Latitudes: {self.lat}')

    def __len__(self):
        return self.data.shape[0] - self.window_size - self.step_size + 1

    def __getitem__(self, idx):
            return self.features[idx : idx + self.window_size], self.targets[idx : idx + self.window_size],self.targets[idx + self.window_size : idx + self.window_size + self.step_size]
    
    def normalize(self, q, t, u, v, w, method = 'std'): 
        if method == 'std':
            q = (q - q.mean()) / q.std()
            t = (t - t.mean()) / t.std()
            u = (u - u.mean()) / u.std()
            v = (v - v.mean()) / v.std()
            w = (w - w.mean()) / w.std()

        return q, t, u, v, w        
    
    def calculate_wind(self, u, v):
        if self.series_target:
            u = u[:, u.shape[1]//2, u.shape[2]//2]
            v = v[:, v.shape[1]//2, v.shape[2]//2]

        self.wspd = np.sqrt(u**2 + v**2)
        self.wdir = np.arctan2(u, v)

    def get_area(self, area: Tuple[int, int]):
        lon = np.argmin(np.abs(self.lon - area[1]))

        lat = np.argmin(np.abs(self.lat - area[0]))

        if self.spaces != 0:

            self.lon = self.lon[lon - self.spaces:lon + self.spaces]
            self.lat = self.lat[lat - self.spaces: lat + self.spaces]

            self.data = self.data[:, lat - self.spaces: lat + self.spaces, lon - self.spaces:lon + self.spaces, :]
        else:

            self.lon = self.lon[lon]
            self.lat = self.lat[lat]

            self.data = self.data[:, lat, lon, :]

    def plot_area(self):
        if self.spaces != 0:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

            ax.coastlines()

            ax.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()])

            lon, lat = self.lon, self.lat
            contour = ax.contourf(lon, lat, self.targets[0].detach().numpy(), transform=ccrs.PlateCarree())

            fig.colorbar(contour, ax=ax, orientation='vertical', label='Wind Speed (m/s)')

            plt.show()
        else:
            print('Cannot plot area with only one point')

    def plot_animation(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plots features and targets from the windowed arrays for visualization.

        Args:
            seed (int): Seed for reproducibility in selecting samples. Default is 0.
            frame_rate (int): The frame rate for the animation. Default is 16.
            levels (int): Number of contour levels for the plot. Default is 10.

        Returns:
            HTML: An HTML object representing the animation.
        """
        if self.spaces != 0:
            bounds = [self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()]

            features = self.features[seed:seed + self.window_size * self.intervals:self.intervals]
            targets = self.targets[seed + self.window_size * self.intervals:seed + (self.window_size + self.step_size) * self.intervals: self.intervals]
            
            time_features = self.times[seed:seed + self.window_size * self.intervals:self.intervals]
            time_targets = self.times[seed + self.window_size * self.intervals:seed + (self.window_size + self.step_size) * self.intervals: self.intervals]

            time_features = pd.to_datetime(time_features)
            time_targets = pd.to_datetime(time_targets)

            fig, axs = plt.subplots(1, 2, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

            vmin = min(features.min().item(), targets.min().item())
            vmax = max(features.max().item(), targets.max().item())

            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

            for ax in axs:
                ax.set_extent(bounds, crs=ccrs.PlateCarree())
                ax.coastlines()

            feat = axs[0].contourf(self.lon, self.lat, features[0, :, :, 2], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
            tar = axs[1].contourf(self.lon, self.lat, targets[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
            axs[1].set_title('Target')

            fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
            fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')

            def animate(i):
                axs[0].clear()
                axs[0].coastlines()

                axs[0].contourf(self.lon, self.lat, features[i, :, :, 2], levels=levels, vmin=vmin, vmax = vmax)

                axs[0].set_title(f'Window {i} - {time_features[i].strftime("%Y-%m-%d %H:%M:%S")}')
                if self.step_size > 1:
                    axs[1].contourf(self.lon, self.lat, targets[i % self.step_size], levels=levels, vmin=vmin, vmax = vmax)
                    axs[1].set_title(f'Target - {time_targets[i % self.step_size].strftime("%Y-%m-%d %H:%M:%S")}')
                # return pcm

                
            frames = features.shape[0]

            interval = 1000 / frame_rate

            ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

            plt.close(fig)

            return HTML(ani.to_jshtml())
        else:
            print('Cannot plot area with only one point')

    def plot_point(self, seed: int = 0):
        if self.spaces == 0:
            plt.figure(figsize=(10, 5))

            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 0], label='Humidity')
            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 1], label='Temperature')
            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 2], label='U-Wind')
            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 3], label='V-Wind')
            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 4], label='W-Wind')
            plt.plot(self.times[seed:seed + self.window_size], self.features[seed:seed + self.window_size, 5], label='Wind Speed', linestyle='--')

            plt.plot(self.times[seed + self.window_size:seed + self.window_size + self.step_size], self.targets[seed + self.window_size:seed + self.window_size + self.step_size], label='Wind Speed', linestyle='--')
            
            plt.xticks(rotation=45)
            plt.legend()
            plt.show()
        else:
            print('Cannot plot point with multiple points')           

# Training model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import wandb

def train_model(model, train_loader, val_loader, n_epochs=50, warmup_epochs=5,
                initial_lr=1e-3, early_stopping_patience=5, 
                checkpoint_path='best_model.pth', device=None):

    # Initialize criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Warmup and Main Scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # Early Stopping Variables
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for i, batch in enumerate(train_loader):
            x, _, y = batch

            if device:
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x.float())
            loss = criterion(y_pred, y.float())
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Print progress every 100 batches
            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')

        # Average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch} Completed, Average Training Loss: {avg_epoch_loss:.4f}')

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, _, val_y in val_loader:
                if device:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                val_pred = model(val_x.float())
                val_loss += criterion(val_pred, val_y.float()).item()
            avg_val_loss = val_loss / len(val_loader)
            print(f'Validation Loss after Epoch {epoch}: {avg_val_loss:.4f}')

        # Step schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(avg_val_loss)

        # Log learning rate and losses to Weights & Biases
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        })
        print(f"Learning rate after Epoch {epoch}: {current_lr:.6f}")

        # Check for model improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model
            torch.save(model.state_dict(), checkpoint_path)
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Early stopping patience counter: {patience_counter}/{early_stopping_patience}')

        # Early stopping condition
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Load the best model after training completes
    model.load_state_dict(torch.load(checkpoint_path))
    print("Best model loaded for evaluation or further use.")

    

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for x,_, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
    return test_loss / len(test_loader)