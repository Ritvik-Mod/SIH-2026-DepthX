/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  webpack: (config) => {
    // geotiff pulls in optional node builtins; stub them for the browser bundle
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false, path: false, http: false, https: false, zlib: false, stream: false,
    };
    return config;
  },
};
export default nextConfig;
