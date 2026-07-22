-- AI Virtual Doctor — MySQL Schema
-- Run this in MySQL: source schema.sql

CREATE DATABASE IF NOT EXISTS ai_doctor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ai_doctor;

CREATE TABLE IF NOT EXISTS users (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    full_name    VARCHAR(100)  NOT NULL,
    email        VARCHAR(150)  NOT NULL UNIQUE,
    password     VARCHAR(255)  NOT NULL,
    created_at   DATETIME      DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT           NOT NULL,
    role         ENUM('user','assistant') NOT NULL,
    message      TEXT          NOT NULL,
    created_at   DATETIME      DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT          NOT NULL,
    symptoms        TEXT         NOT NULL,
    predicted       VARCHAR(200) NOT NULL,
    confidence      FLOAT        NOT NULL,
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
